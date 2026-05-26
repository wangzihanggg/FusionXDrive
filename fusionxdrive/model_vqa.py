"""
Multi-Modal VLM for Driving Scene Understanding

Architecture:
  Image Encoder:  DINOv3-base (frozen)      -> patch features [B, 256, 768]
  LiDAR Encoder:  VoxelNeXt-style           -> global feature [B, 256]
  Radar Encoder:  RadarNeXt-style           -> global feature [B, 256]
  Bridge:         Q-Former                  -> query tokens  [B, 64, 896]
  LLM:            Qwen2.5-0.5B + LoRA      -> text generation

Changes from v1:
  - LLM is no longer fully frozen; LoRA adapters are inserted into
    all attention projection layers (q, k, v, o) in every transformer block.
  - freeze_llm_model flag now controls the BASE weights only; LoRA params
    are always trainable.
  - Config gains lora_r, lora_alpha, lora_dropout fields.
  - _log_param_stats now also shows LoRA parameter counts.
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel, PretrainedConfig,
    AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List

from fusionxdrive.point_cloud_encoders import VoxelNeXtEncoder, RadarNeXtEncoder
from fusionxdrive.qformer import QFormerBridge


# =============================================================================
# Minimal LoRA implementation (no peft dependency)
# =============================================================================

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a LoRA side-path.

    forward(x) = base_linear(x) + (x @ A^T @ B^T) * (alpha / r)
    A is initialised with kaiming_uniform, B with zeros → zero init at start.
    """
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float = 0.05):
        super().__init__()
        self.linear   = linear          # original (frozen) weight
        self.r        = r
        self.scaling  = alpha / r

        in_features  = linear.in_features
        out_features = linear.out_features

        self.lora_A  = nn.Linear(in_features,  r,            bias=False)
        self.lora_B  = nn.Linear(r,            out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # LoRA init: A ~ kaiming, B = 0  → delta-W = 0 at start
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the base linear weight
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + lora

    def extra_repr(self):
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"r={self.r}, scaling={self.scaling:.3f}")


def apply_lora_to_model(model: nn.Module,
                        target_modules: List[str],
                        r: int,
                        alpha: float,
                        dropout: float) -> int:
    """
    Walk `model` and replace every nn.Linear whose name ends with one of
    `target_modules` with a LoRALinear wrapper.

    Returns the number of parameters added by LoRA.
    """
    lora_params = 0
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                # Navigate to parent and replace attribute
                parts  = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                attr   = parts[-1]
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr, lora_layer)
                lora_params += r * (module.in_features + module.out_features)
                break  # avoid double-replacing

    return lora_params


# =============================================================================
# Config
# =============================================================================

class MultiModalVLMConfig(PretrainedConfig):
    model_type = "multimodal_vlm"

    def __init__(self,
                 llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",
                 vision_model_path="facebook/dinov3-base",
                 freeze_vision_model=True,
                 freeze_llm_model=False,      # False=full LLM fine-tune, True=frozen
                 # Q-Former
                 qformer_dim=512,
                 qformer_layers=4,
                 qformer_heads=8,
                 num_query_tokens=64,
                 # Point cloud encoders
                 lidar_output_dim=256,
                 radar_output_dim=256,
                 lidar_pc_range=None,
                 radar_pc_range=None,
                 lidar_pillar_size=None,
                 radar_pillar_size=None,
                 # Misc
                 image_pad_num=64,
                 **kwargs):

        self.llm_model_path      = llm_model_path
        self.vision_model_path   = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_llm_model    = freeze_llm_model

        self.qformer_dim         = qformer_dim
        self.qformer_layers      = qformer_layers
        self.qformer_heads       = qformer_heads
        self.num_query_tokens    = num_query_tokens

        self.lidar_output_dim    = lidar_output_dim
        self.radar_output_dim    = radar_output_dim
        self.lidar_pc_range      = lidar_pc_range  or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.radar_pc_range      = radar_pc_range  or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.lidar_pillar_size   = lidar_pillar_size or [0.2,  0.2,  8.0]
        self.radar_pillar_size   = radar_pillar_size or [0.32, 0.32, 8.0]
        self.image_pad_num       = image_pad_num

        super().__init__(**kwargs)


# =============================================================================
# Model
# =============================================================================

class MultiModalVLM(PreTrainedModel):
    config_class = MultiModalVLMConfig

    def __init__(self, config: MultiModalVLMConfig):
        super().__init__(config)
        self.config = config

        # ------------------------------------------------------------------
        # 1. Vision encoder (DINOv3) — always frozen
        # ------------------------------------------------------------------
        self.vision_model = AutoModel.from_pretrained(config.vision_model_path)
        self.processor    = AutoImageProcessor.from_pretrained(config.vision_model_path)
        vision_hidden_size = self.vision_model.config.hidden_size  # 768

        for param in self.vision_model.parameters():
            param.requires_grad = False

        # ------------------------------------------------------------------
        # 2. LiDAR encoder
        # ------------------------------------------------------------------
        self.lidar_encoder = VoxelNeXtEncoder(
            output_dim=config.lidar_output_dim,
            pc_range=config.lidar_pc_range,
            pillar_size=config.lidar_pillar_size,
        )

        # ------------------------------------------------------------------
        # 3. Radar encoder
        # ------------------------------------------------------------------
        self.radar_encoder = RadarNeXtEncoder(
            output_dim=config.radar_output_dim,
            pc_range=config.radar_pc_range,
            pillar_size=config.radar_pillar_size,
        )

        # ------------------------------------------------------------------
        # 4. LLM (Qwen2.5-0.5B)
        # freeze_llm_model=False → full fine-tuning (all params trainable)
        # freeze_llm_model=True  → frozen base LLM
        # ------------------------------------------------------------------
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
        llm_hidden_size = self.llm_model.config.hidden_size  # 896

        for param in self.llm_model.parameters():
            param.requires_grad = not config.freeze_llm_model

        lora_params = 0

        # ------------------------------------------------------------------
        # 5. Q-Former bridge
        # ------------------------------------------------------------------
        self.qformer = QFormerBridge(
            image_dim=vision_hidden_size,
            lidar_dim=config.lidar_output_dim,
            radar_dim=config.radar_output_dim,
            qformer_dim=config.qformer_dim,
            llm_dim=llm_hidden_size,
            num_query_tokens=config.num_query_tokens,
            num_layers=config.qformer_layers,
            num_heads=config.qformer_heads,
        )

        self._log_param_stats()

    # ------------------------------------------------------------------
    def _log_param_stats(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        llm_mode  = "frozen" if self.config.freeze_llm_model else "full fine-tune"

        print("Model parameter stats:")
        print(f"  Total:        {total:>12,}")
        print(f"  Trainable:    {trainable:>12,}")
        print(f"  Frozen:       {frozen:>12,}")
        print(f"  LLM mode:     {llm_mode}")

        modules = {
            'DINOv3 (vision)':   self.vision_model,
            'VoxelNeXt (lidar)': self.lidar_encoder,
            'RadarNeXt (radar)': self.radar_encoder,
            'Q-Former (bridge)': self.qformer,
            'Qwen2.5 (LLM)':     self.llm_model,
        }
        for name, mod in modules.items():
            n = sum(p.numel() for p in mod.parameters())
            t = sum(p.numel() for p in mod.parameters() if p.requires_grad)
            print(f"  {name:25s}: {n:>12,} total, {t:>12,} trainable")

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------
    def _extract_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs  = self.vision_model(pixel_values)
            features = outputs.last_hidden_state[:, 1:, :]   # drop CLS
        return features

    def _extract_lidar_features(self, lidar_points: List[torch.Tensor]) -> torch.Tensor:
        return self.lidar_encoder(lidar_points)

    def _extract_radar_features(self, radar_points: List[torch.Tensor]) -> torch.Tensor:
        return self.radar_encoder(radar_points)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self,
                input_ids:      torch.Tensor,
                labels:         Optional[torch.Tensor]           = None,
                pixel_values:   Optional[torch.Tensor]           = None,
                lidar_points:   Optional[List[torch.Tensor]]     = None,
                radar_points:   Optional[List[torch.Tensor]]     = None,
                attention_mask: Optional[torch.Tensor]           = None):

        device = input_ids.device
        B      = input_ids.shape[0]

        # 1. Text embeddings
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)  # [B, T, D]

        # 2. Modality features
        if pixel_values is not None:
            image_features = self._extract_image_features(pixel_values)
        else:
            image_features = torch.zeros(
                B, 1, self.vision_model.config.hidden_size, device=device)

        if lidar_points is not None:
            lidar_features = self._extract_lidar_features([p.to(device) for p in lidar_points])
        else:
            lidar_features = torch.zeros(B, self.config.lidar_output_dim, device=device)

        if radar_points is not None:
            radar_features = self._extract_radar_features([p.to(device) for p in radar_points])
        else:
            radar_features = torch.zeros(B, self.config.radar_output_dim, device=device)

        # 3. Q-Former fusion
        query_tokens = self.qformer(
            image_features.to(text_embeds.dtype),
            lidar_features.to(text_embeds.dtype),
            radar_features.to(text_embeds.dtype),
        )  # [B, num_query_tokens, D_llm]

        # 4. Inject query tokens at <|image_pad|> positions
        image_pad_id  = self.tokenizer('<|image_pad|>')['input_ids'][0]
        inputs_embeds = self._merge_multimodal_embeds(
            text_embeds, query_tokens, input_ids, image_pad_id
        )

        # 5. LLM forward (LoRA adapters participate automatically)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs[0]

        # 6. Loss — ignore_index=pad_token_id (same as wild-drive)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1).to(logits.device),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    def _merge_multimodal_embeds(self, text_embeds, query_tokens, input_ids, image_pad_id):
        """Replace <|image_pad|> positions with Q-Former query tokens."""
        B, T, D    = query_tokens.shape
        batch_idx, token_idx = torch.where(input_ids == image_pad_id)

        if len(batch_idx) == 0:
            return text_embeds

        query_flat = query_tokens.reshape(-1, D)
        text_embeds = text_embeds.clone()   # avoid in-place on leaf tensor
        text_embeds[batch_idx, token_idx] = query_flat.to(text_embeds.dtype)
        return text_embeds

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self,
                 pixel_values:  torch.Tensor,
                 lidar_points:  List[torch.Tensor],
                 radar_points:  List[torch.Tensor],
                 prompt_ids:    torch.Tensor,
                 max_new_tokens: int  = 512,
                 temperature:    float = 0.1,
                 **kwargs):

        device = prompt_ids.device
        B      = prompt_ids.shape[0]

        # Modality features
        image_features = self._extract_image_features(pixel_values)
        lidar_features = self._extract_lidar_features([p.to(device) for p in lidar_points])
        radar_features = self._extract_radar_features([p.to(device) for p in radar_points])

        # Build fused embeddings
        text_embeds   = self.llm_model.get_input_embeddings()(prompt_ids)
        query_tokens  = self.qformer(
            image_features.to(text_embeds.dtype),
            lidar_features.to(text_embeds.dtype),
            radar_features.to(text_embeds.dtype),
        )
        image_pad_id  = self.tokenizer('<|image_pad|>')['input_ids'][0]
        inputs_embeds = self._merge_multimodal_embeds(
            text_embeds, query_tokens, prompt_ids, image_pad_id
        )

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        return outputs


if __name__ == '__main__':
    print("Initializing MultiModalVLM (full LLM fine-tuning)...")
    config = MultiModalVLMConfig(freeze_llm_model=False)
    print(f"freeze_llm_model={config.freeze_llm_model}")