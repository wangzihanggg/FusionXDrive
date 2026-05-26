# """
# Multi-Modal VLM + Trajectory Planning.

# Planning condition = Q-Former tokens (scene info) + LLM plan_pad hidden states (language context).
# Stage 2: LLM frozen, QFormer frozen, only planner trainable.
# """
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# import math, torch, torch.nn as nn, torch.nn.functional as F
# from transformers import (PreTrainedModel, PretrainedConfig, AutoModel,
#                           AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor)
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from typing import Optional, List
# from point_cloud_encoders import VoxelNeXtEncoder, RadarNeXtEncoder
# from qformer import QFormerBridge
# from diffusion_planner import TruncatedDiffusionPlanner

# class LoRALinear(nn.Module):
#     def __init__(self, linear, r, alpha, dropout=0.05):
#         super().__init__()
#         self.linear, self.r, self.scaling = linear, r, alpha / r
#         self.lora_A = nn.Linear(linear.in_features, r, bias=False)
#         self.lora_B = nn.Linear(r, linear.out_features, bias=False)
#         self.dropout = nn.Dropout(p=dropout)
#         nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B.weight)
#         self.linear.weight.requires_grad_(False)
#         if self.linear.bias is not None:
#             self.linear.bias.requires_grad_(False)
#     def forward(self, x):
#         return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

# def apply_lora_to_model(model, target_modules, r, alpha, dropout):
#     lora_params = 0
#     for name, module in list(model.named_modules()):
#         for target in target_modules:
#             if name.endswith(target) and isinstance(module, nn.Linear):
#                 parts = name.split('.')
#                 parent = model
#                 for part in parts[:-1]:
#                     parent = getattr(parent, part)
#                 setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
#                 lora_params += r * (module.in_features + module.out_features)
#                 break
#     return lora_params

# class MultiModalVLMConfig(PretrainedConfig):
#     model_type = "multimodal_vlm"
#     def __init__(self, llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",
#                  vision_model_path="facebook/dinov3-base",
#                  freeze_vision_model=True, freeze_llm_model=False,
#                  use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05,
#                  lora_target_modules=None,
#                  qformer_dim=512, qformer_layers=4, qformer_heads=8, num_query_tokens=64,
#                  lidar_output_dim=256, radar_output_dim=256,
#                  lidar_pc_range=None, radar_pc_range=None,
#                  lidar_pillar_size=None, radar_pillar_size=None,
#                  image_pad_num=64,
#                  use_planning=True, num_planning_tokens=4,
#                  planning_num_waypoints=8, planning_waypoint_dim=3,
#                  planning_num_anchors=8, planning_cond_dim=256,
#                  planning_denoise_hidden=256, planning_denoise_blocks=4,
#                  planning_t_trunc=5, planning_n_infer_steps=2,
#                  planning_loss_weight=1.0, **kwargs):
#         self.llm_model_path = llm_model_path
#         self.vision_model_path = vision_model_path
#         self.freeze_vision_model = freeze_vision_model
#         self.freeze_llm_model = freeze_llm_model
#         self.use_lora = use_lora; self.lora_r = lora_r; self.lora_alpha = lora_alpha
#         self.lora_dropout = lora_dropout
#         self.lora_target_modules = lora_target_modules or ['q_proj','k_proj','v_proj','o_proj']
#         self.qformer_dim = qformer_dim; self.qformer_layers = qformer_layers
#         self.qformer_heads = qformer_heads; self.num_query_tokens = num_query_tokens
#         self.lidar_output_dim = lidar_output_dim; self.radar_output_dim = radar_output_dim
#         self.lidar_pc_range = lidar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
#         self.radar_pc_range = radar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
#         self.lidar_pillar_size = lidar_pillar_size or [0.2,0.2,8.0]
#         self.radar_pillar_size = radar_pillar_size or [0.32,0.32,8.0]
#         self.image_pad_num = image_pad_num
#         self.use_planning = use_planning; self.num_planning_tokens = num_planning_tokens
#         self.planning_num_waypoints = planning_num_waypoints
#         self.planning_waypoint_dim = planning_waypoint_dim
#         self.planning_num_anchors = planning_num_anchors
#         self.planning_cond_dim = planning_cond_dim
#         self.planning_denoise_hidden = planning_denoise_hidden
#         self.planning_denoise_blocks = planning_denoise_blocks
#         self.planning_t_trunc = planning_t_trunc
#         self.planning_n_infer_steps = planning_n_infer_steps
#         self.planning_loss_weight = planning_loss_weight
#         super().__init__(**kwargs)

# class MultiModalVLM(PreTrainedModel):
#     config_class = MultiModalVLMConfig
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#         # 1. Vision (frozen)
#         self.vision_model = AutoModel.from_pretrained(config.vision_model_path)
#         self.processor = AutoImageProcessor.from_pretrained(config.vision_model_path)
#         vh = self.vision_model.config.hidden_size
#         for p in self.vision_model.parameters(): p.requires_grad = False
#         # 2. LiDAR
#         self.lidar_encoder = VoxelNeXtEncoder(output_dim=config.lidar_output_dim,
#             pc_range=config.lidar_pc_range, pillar_size=config.lidar_pillar_size)
#         # 3. Radar
#         self.radar_encoder = RadarNeXtEncoder(output_dim=config.radar_output_dim,
#             pc_range=config.radar_pc_range, pillar_size=config.radar_pillar_size)
#         # 4. LLM
#         self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
#         lh = self.llm_model.config.hidden_size
#         for p in self.llm_model.parameters():
#             p.requires_grad = not config.freeze_llm_model
#         lora_n = 0
#         if config.use_lora and config.freeze_llm_model:
#             lora_n = apply_lora_to_model(self.llm_model,
#                 config.lora_target_modules, config.lora_r, config.lora_alpha, config.lora_dropout)
#             print(f"[LoRA] r={config.lora_r} alpha={config.lora_alpha} → {lora_n:,} params")
#         # 5. Q-Former
#         self.qformer = QFormerBridge(image_dim=vh, lidar_dim=config.lidar_output_dim,
#             radar_dim=config.radar_output_dim, qformer_dim=config.qformer_dim,
#             llm_dim=lh, num_query_tokens=config.num_query_tokens,
#             num_layers=config.qformer_layers, num_heads=config.qformer_heads)
#         # 6. Planner — condition = QFormer tokens + LLM plan_pad hidden states
#         if config.use_planning:
#             # Planner receives BOTH sources concatenated:
#             #   QFormer: num_query_tokens tokens × lh dim (scene perception)
#             #   LLM:     num_planning_tokens tokens × lh dim (language context)
#             # Total tokens for condition = num_query_tokens + num_planning_tokens
#             total_cond_tokens = config.num_query_tokens + config.num_planning_tokens
#             self.planner = TruncatedDiffusionPlanner(
#                 llm_hidden_dim=lh,
#                 num_planning_tokens=total_cond_tokens,  # combined token count
#                 num_waypoints=config.planning_num_waypoints,
#                 waypoint_dim=config.planning_waypoint_dim,
#                 num_anchors=config.planning_num_anchors, cond_dim=config.planning_cond_dim,
#                 denoise_hidden=config.planning_denoise_hidden,
#                 denoise_blocks=config.planning_denoise_blocks,
#                 t_trunc=config.planning_t_trunc, n_infer_steps=config.planning_n_infer_steps)
#             print(f"[Planner] condition = QFormer {config.num_query_tokens} tokens "
#                   f"+ LLM {config.num_planning_tokens} plan tokens "
#                   f"= {total_cond_tokens} tokens × {lh}d")
#         self._log_stats(lora_n)

#     def _log_stats(self, lora_n=0):
#         tot = sum(p.numel() for p in self.parameters())
#         tra = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         mode = "frozen+LoRA" if (self.config.freeze_llm_model and self.config.use_lora) \
#                else ("frozen" if self.config.freeze_llm_model else "full-ft")
#         print(f"Params: total={tot:,} trainable={tra:,} frozen={tot-tra:,} LLM={mode}")
#         if lora_n: print(f"  LoRA params: {lora_n:,}")
#         for n, m in [('Vision',self.vision_model),('LiDAR',self.lidar_encoder),
#                      ('Radar',self.radar_encoder),('QFormer',self.qformer),('LLM',self.llm_model)]:
#             t = sum(p.numel() for p in m.parameters())
#             tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
#             print(f"  {n:12s}: {t:>12,} total {tr:>12,} trainable")
#         if self.config.use_planning:
#             t = sum(p.numel() for p in self.planner.parameters())
#             print(f"  {'Planner':12s}: {t:>12,} total {t:>12,} trainable")

#     def _img_feat(self, pv):
#         with torch.no_grad(): return self.vision_model(pv).last_hidden_state[:,1:,:]

#     def forward(self, input_ids, labels=None, pixel_values=None, lidar_points=None,
#                 radar_points=None, attention_mask=None, gt_waypoints=None):
#         dev, B = input_ids.device, input_ids.shape[0]
#         te = self.llm_model.get_input_embeddings()(input_ids)
#         imgf = self._img_feat(pixel_values) if pixel_values is not None \
#                else torch.zeros(B,1,self.vision_model.config.hidden_size,device=dev)
#         lidf = self.lidar_encoder([p.to(dev) for p in lidar_points]) if lidar_points \
#                else torch.zeros(B,self.config.lidar_output_dim,device=dev)
#         radf = self.radar_encoder([p.to(dev) for p in radar_points]) if radar_points \
#                else torch.zeros(B,self.config.radar_output_dim,device=dev)

#         # Q-Former: fused multi-modal tokens
#         qt = self.qformer(imgf.to(te.dtype), lidf.to(te.dtype), radf.to(te.dtype))
#         # qt: [B, num_query_tokens, D] — rich scene representation

#         pid = self.tokenizer('<|image_pad|>')['input_ids'][0]
#         ie = self._merge(te, qt, input_ids, pid)
#         need_h = self.config.use_planning and gt_waypoints is not None
#         out = self.llm_model(inputs_embeds=ie, attention_mask=attention_mask,
#                              output_hidden_states=need_h)
#         logits = out[0]

#         # LM loss
#         lm_loss = None
#         if labels is not None:
#             lm_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
#                 logits.view(-1, logits.size(-1)), labels.view(-1).to(dev))

#         # Planning loss
#         plan_loss = None
#         if self.config.use_planning and gt_waypoints is not None:
#             ptid = self.tokenizer('<|plan_pad|>')['input_ids'][0]
#             bi, si = torch.where(input_ids == ptid)
#             np_ = self.config.num_planning_tokens
#             if len(bi) == B * np_:
#                 hs = out.hidden_states[-1]
#                 plan_hidden = hs[bi, si].view(B, np_, -1)  # [B, num_plan, D]

#                 # Combine: QFormer tokens (scene) + LLM plan tokens (language)
#                 # qt: [B, 64, D], plan_hidden: [B, num_plan, D]
#                 combined = torch.cat([qt.detach(), plan_hidden], dim=1)  # [B, 64+num_plan, D]

#                 po = self.planner(combined, gt_waypoints)
#                 plan_loss = po['loss']
#             else:
#                 print(f"[WARN] plan token mismatch: {len(bi)} vs {B*np_}")

#         loss = lm_loss
#         if plan_loss is not None and loss is not None:
#             loss = loss + self.config.planning_loss_weight * plan_loss
#         elif plan_loss is not None:
#             loss = plan_loss
#         return CausalLMOutputWithPast(loss=loss, logits=logits)

#     def _merge(self, te, qt, ids, pid):
#         B,T,D = qt.shape
#         bi, ti = torch.where(ids == pid)
#         if len(bi)==0: return te
#         te = te.clone()
#         te[bi,ti] = qt.reshape(-1,D).to(te.dtype)
#         return te

#     @torch.no_grad()
#     def generate(self, pixel_values, lidar_points, radar_points, prompt_ids,
#                  max_new_tokens=512, temperature=0.1, return_planning=False, **kw):
#         dev, B = prompt_ids.device, prompt_ids.shape[0]
#         imgf = self._img_feat(pixel_values)
#         lidf = self.lidar_encoder([p.to(dev) for p in lidar_points])
#         radf = self.radar_encoder([p.to(dev) for p in radar_points])
#         te = self.llm_model.get_input_embeddings()(prompt_ids)
#         qt = self.qformer(imgf.to(te.dtype), lidf.to(te.dtype), radf.to(te.dtype))
#         pid = self.tokenizer('<|image_pad|>')['input_ids'][0]
#         ie = self._merge(te, qt, prompt_ids, pid)

#         # Text generation
#         tids = self.llm_model.generate(inputs_embeds=ie, max_new_tokens=max_new_tokens,
#             temperature=temperature, do_sample=temperature>0,
#             pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **kw)

#         if not return_planning or not self.config.use_planning:
#             return tids

#         # ═══ Planning: reconstruct FULL context matching training ═══
#         # Training sees:  [prompt_embeds] [answer_embeds] [plan_pad_embeds]
#         # We must replicate this at inference!

#         # 1. Get embeddings for generated answer tokens
#         answer_embeds = self.llm_model.get_input_embeddings()(tids)  # [B, gen_len, D]

#         # 2. Get plan_pad embeddings
#         ptid = self.tokenizer('<|plan_pad|>')['input_ids'][0]
#         pids = torch.full((B, self.config.num_planning_tokens), ptid, dtype=torch.long, device=dev)
#         plan_embeds = self.llm_model.get_input_embeddings()(pids)

#         # 3. Full context: prompt + answer + plan_pad (matches training!)
#         full_embeds = torch.cat([ie, answer_embeds, plan_embeds], dim=1)
#         po = self.llm_model(inputs_embeds=full_embeds, output_hidden_states=True)
#         plan_hidden = po.hidden_states[-1][:, -self.config.num_planning_tokens:, :]

#         # Combine QFormer + LLM plan tokens
#         combined = torch.cat([qt, plan_hidden], dim=1)  # [B, 64+num_plan, D]

#         planner_out = self.planner(combined, gt_waypoints=None)
#         return tids, planner_out

"""
Multi-Modal VLM + Trajectory Planning.

Planning condition = Q-Former tokens (scene info) + LLM plan_pad hidden states (language context).
Stage 2: LLM frozen, QFormer frozen, only planner trainable.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import math, torch, torch.nn as nn, torch.nn.functional as F
from transformers import (PreTrainedModel, PretrainedConfig, AutoModel,
                          AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List
from fusionxdrive.point_cloud_encoders import VoxelNeXtEncoder, RadarNeXtEncoder
from fusionxdrive.moro_former import MoRoFormer as QFormerBridge
from fusionxdrive.diffusion_planner import TruncatedDiffusionPlanner

class LoRALinear(nn.Module):
    def __init__(self, linear, r, alpha, dropout=0.05):
        super().__init__()
        self.linear, self.r, self.scaling = linear, r, alpha / r
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def apply_lora_to_model(model, target_modules, r, alpha, dropout):
    lora_params = 0
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
                lora_params += r * (module.in_features + module.out_features)
                break
    return lora_params

class MultiModalVLMConfig(PretrainedConfig):
    model_type = "multimodal_vlm"
    def __init__(self, llm_model_path="Qwen/Qwen2.5-0.5B-Instruct",
                 vision_model_path="facebook/dinov3-base",
                 freeze_vision_model=True, freeze_llm_model=False,
                 use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                 lora_target_modules=None,
                 qformer_dim=512, qformer_layers=4, qformer_heads=8, num_query_tokens=64,
                 lidar_output_dim=256, radar_output_dim=256,
                 lidar_pc_range=None, radar_pc_range=None,
                 lidar_pillar_size=None, radar_pillar_size=None,
                 image_pad_num=64,
                 use_planning=True, num_planning_tokens=4,
                 planning_num_waypoints=8, planning_waypoint_dim=3,
                 planning_num_anchors=8, planning_cond_dim=256,
                 planning_denoise_hidden=256, planning_denoise_blocks=4,
                 planning_t_trunc=5, planning_n_infer_steps=2,
                 planning_loss_weight=1.0, **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.freeze_llm_model = freeze_llm_model
        self.use_lora = use_lora; self.lora_r = lora_r; self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ['q_proj','k_proj','v_proj','o_proj']
        self.qformer_dim = qformer_dim; self.qformer_layers = qformer_layers
        self.qformer_heads = qformer_heads; self.num_query_tokens = num_query_tokens
        self.lidar_output_dim = lidar_output_dim; self.radar_output_dim = radar_output_dim
        self.lidar_pc_range = lidar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
        self.radar_pc_range = radar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
        self.lidar_pillar_size = lidar_pillar_size or [0.2,0.2,8.0]
        self.radar_pillar_size = radar_pillar_size or [0.32,0.32,8.0]
        self.image_pad_num = image_pad_num
        self.use_planning = use_planning; self.num_planning_tokens = num_planning_tokens
        self.planning_num_waypoints = planning_num_waypoints
        self.planning_waypoint_dim = planning_waypoint_dim
        self.planning_num_anchors = planning_num_anchors
        self.planning_cond_dim = planning_cond_dim
        self.planning_denoise_hidden = planning_denoise_hidden
        self.planning_denoise_blocks = planning_denoise_blocks
        self.planning_t_trunc = planning_t_trunc
        self.planning_n_infer_steps = planning_n_infer_steps
        self.planning_loss_weight = planning_loss_weight
        super().__init__(**kwargs)

class MultiModalVLM(PreTrainedModel):
    config_class = MultiModalVLMConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 1. Vision (frozen)
        self.vision_model = AutoModel.from_pretrained(config.vision_model_path)
        self.processor = AutoImageProcessor.from_pretrained(config.vision_model_path)
        vh = self.vision_model.config.hidden_size
        for p in self.vision_model.parameters(): p.requires_grad = False
        # 2. LiDAR
        self.lidar_encoder = VoxelNeXtEncoder(output_dim=config.lidar_output_dim,
            pc_range=config.lidar_pc_range, pillar_size=config.lidar_pillar_size)
        # 3. Radar
        self.radar_encoder = RadarNeXtEncoder(output_dim=config.radar_output_dim,
            pc_range=config.radar_pc_range, pillar_size=config.radar_pillar_size)
        # 4. LLM
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
        lh = self.llm_model.config.hidden_size
        for p in self.llm_model.parameters():
            p.requires_grad = not config.freeze_llm_model
        lora_n = 0
        if config.use_lora and config.freeze_llm_model:
            lora_n = apply_lora_to_model(self.llm_model,
                config.lora_target_modules, config.lora_r, config.lora_alpha, config.lora_dropout)
            print(f"[LoRA] r={config.lora_r} alpha={config.lora_alpha} → {lora_n:,} params")
        # 5. Q-Former
        self.qformer = QFormerBridge(image_dim=vh, lidar_dim=config.lidar_output_dim,
            radar_dim=config.radar_output_dim, qformer_dim=config.qformer_dim,
            llm_dim=lh, num_query_tokens=config.num_query_tokens,
            num_layers=config.qformer_layers, num_heads=config.qformer_heads)
        # 6. Planner — condition = QFormer tokens + LLM plan_pad hidden states
        if config.use_planning:
            # Planner receives BOTH sources concatenated:
            #   QFormer: num_query_tokens tokens × lh dim (scene perception)
            #   LLM:     num_planning_tokens tokens × lh dim (language context)
            # Total tokens for condition = num_query_tokens + num_planning_tokens
            total_cond_tokens = config.num_query_tokens + config.num_planning_tokens
            self.planner = TruncatedDiffusionPlanner(
                llm_hidden_dim=lh,
                num_planning_tokens=total_cond_tokens,  # combined token count
                num_waypoints=config.planning_num_waypoints,
                waypoint_dim=config.planning_waypoint_dim,
                num_anchors=config.planning_num_anchors, cond_dim=config.planning_cond_dim,
                denoise_hidden=config.planning_denoise_hidden,
                denoise_blocks=config.planning_denoise_blocks,
                t_trunc=config.planning_t_trunc, n_infer_steps=config.planning_n_infer_steps)
            print(f"[Planner] condition = QFormer {config.num_query_tokens} tokens "
                  f"+ LLM {config.num_planning_tokens} plan tokens "
                  f"= {total_cond_tokens} tokens × {lh}d")
        self._log_stats(lora_n)

    def _log_stats(self, lora_n=0):
        tot = sum(p.numel() for p in self.parameters())
        tra = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode = "frozen+LoRA" if (self.config.freeze_llm_model and self.config.use_lora) \
               else ("frozen" if self.config.freeze_llm_model else "full-ft")
        print(f"Params: total={tot:,} trainable={tra:,} frozen={tot-tra:,} LLM={mode}")
        if lora_n: print(f"  LoRA params: {lora_n:,}")
        for n, m in [('Vision',self.vision_model),('LiDAR',self.lidar_encoder),
                     ('Radar',self.radar_encoder),('QFormer',self.qformer),('LLM',self.llm_model)]:
            t = sum(p.numel() for p in m.parameters())
            tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  {n:12s}: {t:>12,} total {tr:>12,} trainable")
        if self.config.use_planning:
            t = sum(p.numel() for p in self.planner.parameters())
            print(f"  {'Planner':12s}: {t:>12,} total {t:>12,} trainable")

    def _img_feat(self, pv):
        with torch.no_grad(): return self.vision_model(pv).last_hidden_state[:,1:,:]

    def forward(self, input_ids, labels=None, pixel_values=None, lidar_points=None,
                radar_points=None, attention_mask=None, gt_waypoints=None):
        dev, B = input_ids.device, input_ids.shape[0]
        te = self.llm_model.get_input_embeddings()(input_ids)
        imgf = self._img_feat(pixel_values) if pixel_values is not None \
               else torch.zeros(B,1,self.vision_model.config.hidden_size,device=dev)
        lidf = self.lidar_encoder([p.to(dev) for p in lidar_points]) if lidar_points \
               else torch.zeros(B,self.config.lidar_output_dim,device=dev)
        radf = self.radar_encoder([p.to(dev) for p in radar_points]) if radar_points \
               else torch.zeros(B,self.config.radar_output_dim,device=dev)

        # Q-Former: fused multi-modal tokens
        qt = self.qformer(imgf.to(te.dtype), lidf.to(te.dtype), radf.to(te.dtype))
        # qt: [B, num_query_tokens, D] — rich scene representation

        pid = self.tokenizer('<|image_pad|>')['input_ids'][0]
        ie = self._merge(te, qt, input_ids, pid)
        need_h = self.config.use_planning and gt_waypoints is not None
        out = self.llm_model(inputs_embeds=ie, attention_mask=attention_mask,
                             output_hidden_states=need_h)
        logits = out[0]

        # LM loss
        lm_loss = None
        if labels is not None:
            lm_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(dev))

        # Planning loss
        plan_loss = None
        if self.config.use_planning and gt_waypoints is not None:
            ptid = self.tokenizer('<|plan_pad|>')['input_ids'][0]
            bi, si = torch.where(input_ids == ptid)
            np_ = self.config.num_planning_tokens
            if len(bi) == B * np_:
                hs = out.hidden_states[-1]
                plan_hidden = hs[bi, si].view(B, np_, -1)  # [B, num_plan, D]

                # Combine: QFormer tokens (scene) + LLM plan tokens (language)
                # qt: [B, 64, D], plan_hidden: [B, num_plan, D]
                combined = torch.cat([qt.detach(), plan_hidden], dim=1)  # [B, 64+num_plan, D]

                po = self.planner(combined, gt_waypoints)
                plan_loss = po['loss']
            else:
                print(f"[WARN] plan token mismatch: {len(bi)} vs {B*np_}")

        loss = lm_loss
        if plan_loss is not None and loss is not None:
            loss = loss + self.config.planning_loss_weight * plan_loss
        elif plan_loss is not None:
            loss = plan_loss
        # Add MoRo-Former router load-balancing loss
        if self.training and hasattr(self.qformer, 'get_router_loss'):
            router_loss = self.qformer.get_router_loss()
            if loss is not None:
                loss = loss + router_loss
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def _merge(self, te, qt, ids, pid):
        B,T,D = qt.shape
        bi, ti = torch.where(ids == pid)
        if len(bi)==0: return te
        te = te.clone()
        te[bi,ti] = qt.reshape(-1,D).to(te.dtype)
        return te

    @torch.no_grad()
    def generate(self, pixel_values, lidar_points, radar_points, prompt_ids,
                 max_new_tokens=512, temperature=0.1, return_planning=False, **kw):
        dev, B = prompt_ids.device, prompt_ids.shape[0]
        imgf = self._img_feat(pixel_values)
        lidf = self.lidar_encoder([p.to(dev) for p in lidar_points])
        radf = self.radar_encoder([p.to(dev) for p in radar_points])
        te = self.llm_model.get_input_embeddings()(prompt_ids)
        qt = self.qformer(imgf.to(te.dtype), lidf.to(te.dtype), radf.to(te.dtype))
        pid = self.tokenizer('<|image_pad|>')['input_ids'][0]
        ie = self._merge(te, qt, prompt_ids, pid)

        # Text generation
        tids = self.llm_model.generate(inputs_embeds=ie, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=temperature>0,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **kw)

        if not return_planning or not self.config.use_planning:
            return tids

        # ═══ Planning: reconstruct FULL context matching training ═══
        # Training sees:  [prompt_embeds] [answer_embeds] [plan_pad_embeds]
        # We must replicate this at inference!

        # 1. Get embeddings for generated answer tokens
        answer_embeds = self.llm_model.get_input_embeddings()(tids)  # [B, gen_len, D]

        # 2. Get plan_pad embeddings
        ptid = self.tokenizer('<|plan_pad|>')['input_ids'][0]
        pids = torch.full((B, self.config.num_planning_tokens), ptid, dtype=torch.long, device=dev)
        plan_embeds = self.llm_model.get_input_embeddings()(pids)

        # 3. Full context: prompt + answer + plan_pad (matches training!)
        full_embeds = torch.cat([ie, answer_embeds, plan_embeds], dim=1)
        po = self.llm_model(inputs_embeds=full_embeds, output_hidden_states=True)
        plan_hidden = po.hidden_states[-1][:, -self.config.num_planning_tokens:, :]

        # Combine QFormer + LLM plan tokens
        combined = torch.cat([qt, plan_hidden], dim=1)  # [B, 64+num_plan, D]

        planner_out = self.planner(combined, gt_waypoints=None)
        return tids, planner_out