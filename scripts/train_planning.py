# """
# 3-Stage Training for VLM + Truncated Diffusion Planning with LoRA.

# Stage 1: VQA only (LoRA + QFormer + encoders)
#     python train_with_planning.py --stage 1 \
#         --data_root ./UNISCP --output_dir save/stage1_vqa_lora \
#         --num_epochs 10 --sequences RURAL_A0 RURAL_A1 RURAL_A2 RURAL_B0 RURAL_B1 RURAL_B2

# Stage 2: Planning only (freeze VQA, train planner)
#     python train_with_planning.py --stage 2 \
#         --data_root ./UNISCP --output_dir save/stage2_planning \
#         --vqa_checkpoint save/stage1_vqa_lora/final_model \
#         --num_epochs 10 --planning_loss_weight 2.0

# Stage 3 (future): RL fine-tuning with GRPO
# """
# import os, sys, json, argparse, logging
# from pathlib import Path
# from typing import Dict, List, Optional
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# import numpy as np, torch
# from torch.utils.data import Dataset
# from PIL import Image
# from transformers import AutoTokenizer, AutoImageProcessor, TrainingArguments, Trainer

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from model_with_planning import MultiModalVLM, MultiModalVLMConfig
# from dataset import (read_pcd, get_rural_calibration, load_timestamps, find_nearest_idx,
#                      SYSTEM_PROMPT, USER_PROMPT)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# DENSE_LABELS = ["t+0.5s","t+1.0s","t+1.5s","t+2.0s","t+2.5s","t+3.0s","t+3.5s","t+4.0s"]

# def load_waypoints(path, n=8):
#     """Load dense 8-point waypoints from JSON."""
#     wp = np.zeros((n,3), dtype=np.float32)
#     try:
#         with open(path) as f: data = json.load(f)
#         d = {w['label']:w for w in data.get('waypoints',[])}
#         for i, lab in enumerate(DENSE_LABELS[:n]):
#             if lab in d and d[lab].get('available',True):
#                 wp[i] = [d[lab]['x'], d[lab]['y'], d[lab].get('z',0.0)]
#     except: pass
#     return wp

# class UniscpDatasetWithPlanning(Dataset):
#     RURAL = ['RURAL_A0','RURAL_A1','RURAL_A2','RURAL_B0','RURAL_B1','RURAL_B2']
#     OTHER = ['FENDUAN_1','KUNSHAN_LUCE6','NIGHT_GAOJIAOQIAO','CP_MSCLIKE','GARDEN_MSCLIKE','LOOP1_MSCLIKE']

#     def __init__(self, data_root, sequences=None, tokenizer=None, processor=None,
#                  image_pad_num=64, max_lidar=40000, max_radar=16000,
#                  lidar_pc_range=None, radar_pc_range=None,
#                  num_planning_tokens=4, use_planning=True):
#         super().__init__()
#         self.root = Path(data_root); self.tok = tokenizer; self.proc = processor
#         self.ipad = image_pad_num; self.ml = max_lidar; self.mr = max_radar
#         self.npt = num_planning_tokens; self.up = use_planning
#         self.lpr = lidar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
#         self.rpr = radar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
#         self.calib = get_rural_calibration()
#         if sequences is None: sequences = self.RURAL + self.OTHER
#         self.samples = []; nw = 0
#         for sq in sequences:
#             d = self.root / sq
#             if d.exists(): nw += self._idx(sq, d)
#         logger.info(f"Loaded {len(self.samples)} samples, {nw} with waypoints")

#     def _fsd(self, sd, p):
#         t = sd / p
#         if not t.exists(): return None
#         n = t / p
#         return n if n.exists() else t

#     def _idx(self, sn, sd):
#         imd = self._fsd(sd,'1_IMAGE'); lid = self._fsd(sd,'2_LIDAR')
#         rad = self._fsd(sd,'3_RADAR'); cd = sd/'6_CAPTION'; wd = sd/'7_PLANNING'/'WAYPOINTS'
#         if not all([imd,lid,rad]) or not cd.exists(): return 0
#         tfs = [imd/'timestamp_image_left.txt', lid/'timestamp_lidar.txt', rad/'timestamp_radar.txt']
#         if not all(f.exists() for f in tfs): return 0
#         tsi = load_timestamps(str(tfs[0]))
#         tsl = load_timestamps(str(tfs[1]))
#         tsr = load_timestamps(str(tfs[2]))
#         ld = imd/'LEFT'; lpd = lid/'PCD'; rpd = rad/'PCD'
#         if not all(d.exists() for d in [ld,lpd,rpd]): return 0
#         c = 0
#         for ix, ts in tsi:
#             ip = ld/f"{ix}.png"; cp = cd/f"{ix}.json"
#             if not ip.exists() or not cp.exists(): continue
#             li = find_nearest_idx(ts, tsl, max_diff=0.15)
#             ri = find_nearest_idx(ts, tsr, max_diff=0.15)
#             if li is None or ri is None: continue
#             lp = lpd/f"{li}.pcd"; rp = rpd/f"{ri}.pcd"
#             if not lp.exists() or not rp.exists(): continue
#             wp = wd/f"{ix}.json"; hw = wp.exists()
#             if hw: c += 1
#             self.samples.append({'seq':sn,'img_idx':ix,'img_path':str(ip),
#                 'lidar_path':str(lp),'radar_path':str(rp),'caption_path':str(cp),
#                 'waypoint_path':str(wp) if hw else None})
#         return c

#     def __len__(self): return len(self.samples)

#     def _ll(self, p):
#         try:
#             d = read_pcd(p, ['x','y','z','intensity'])
#             x,y,z = d.get('x',np.zeros(0)),d.get('y',np.zeros(0)),d.get('z',np.zeros(0))
#             i = d.get('intensity',np.zeros_like(x))
#             if len(x)==0: return np.zeros((0,4),dtype=np.float32)
#             pts = np.stack([x,y,z,i],-1).astype(np.float32)
#             pts = pts[np.all(np.isfinite(pts),-1)]; pts = pts[np.any(pts[:,:3]!=0,-1)]
#             return pts
#         except: return np.zeros((0,4),dtype=np.float32)

#     def _lr(self, p):
#         try:
#             d = read_pcd(p, ['x','y','z','doppler','power','recoveredSpeed'])
#             x,y,z = d.get('x',np.zeros(0)),d.get('y',np.zeros(0)),d.get('z',np.zeros(0))
#             dp,pw,sp = d.get('doppler',np.zeros_like(x)),d.get('power',np.zeros_like(x)),d.get('recoveredSpeed',np.zeros_like(x))
#             if len(x)==0: return np.zeros((0,6),dtype=np.float32)
#             pts = np.stack([x,y,z,dp,pw,sp],-1).astype(np.float32)
#             return pts[np.all(np.isfinite(pts),-1)]
#         except: return np.zeros((0,6),dtype=np.float32)

#     @staticmethod
#     def _sub(pts, mx):
#         if len(pts)==0: return np.zeros((1,pts.shape[1] if pts.ndim>1 else 4),dtype=np.float32)
#         return pts[np.random.choice(len(pts),mx,replace=False)] if len(pts)>mx else pts

#     def __getitem__(self, i):
#         s = self.samples[i]
#         try: img = Image.open(s['img_path']).convert('RGB')
#         except: img = Image.new('RGB',(720,540),'black')
#         lidar = self._ll(s['lidar_path'])
#         if len(lidar)>0: lidar = self.calib.crop_lidar_to_fov(lidar)
#         lidar = self._sub(lidar, self.ml)
#         radar = self._lr(s['radar_path'])
#         if len(radar)>0: radar = self.calib.crop_radar_to_fov(radar)
#         radar = self._sub(radar, self.mr)
#         try:
#             with open(s['caption_path']) as f: cap = json.load(f)
#         except:
#             cap = {"weather":{"condition":"unknown","illumination":"unknown"},
#                    "traffic_light":{"present":"unknown","state":"unknown"},
#                    "traffic_sign":{"present":"unknown","category":"unknown"},
#                    "participants":{"count":"0","objects":[]},
#                    "hazard_region":{"present":"unknown","type":"unknown","direction":"unknown"},
#                    "forward_drivability":{"status":"unknown"},
#                    "lane_keeping":{"status":"unknown","deviation":"unknown"},
#                    "driving_advice":{"action":"unknown"},"explanation":{"reason":"unknown"}}

#         gwp = np.zeros((8,3),dtype=np.float32)
#         if self.up and s.get('waypoint_path'):
#             gwp = load_waypoints(s['waypoint_path'])

#         pv = self.proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
#         ans = json.dumps(cap, ensure_ascii=False)
#         msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"<image>\n"+USER_PROMPT}]
#         qt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
#         qt = qt.replace('<image>', '<|image_pad|>'*self.ipad)
#         qi = self.tok(qt)["input_ids"]
#         ai = self.tok(ans + self.tok.eos_token)["input_ids"]

#         # Only append plan tokens if planning is enabled
#         if self.up and self.npt > 0:
#             pi = self.tok('<|plan_pad|>'*(self.npt+1))["input_ids"]
#         else:
#             pi = []

#         all_ids = qi + ai + pi
#         all_lab = [self.tok.pad_token_id]*len(qi) + ai + [self.tok.pad_token_id]*len(pi)

#         result = {"input_ids":all_ids[:-1], "labels":all_lab[1:], "pixel_values":pv,
#                   "lidar_points":lidar.astype(np.float32), "radar_points":radar.astype(np.float32),
#                   "n_lidar":len(lidar), "n_radar":len(radar)}

#         if self.up:
#             result["gt_waypoints"] = gwp

#         return result

# class Collator:
#     def __init__(self, tok, plan=True): self.tok=tok; self.plan=plan
#     def __call__(self, fs):
#         mx = max(len(f["input_ids"]) for f in fs)
#         ids,lab,pv,li,ra,wp = [],[],[],[],[],[]
#         for f in fs:
#             pd = mx-len(f["input_ids"])
#             ids.append(f["input_ids"]+[self.tok.pad_token_id]*pd)
#             lab.append(f["labels"]+[self.tok.pad_token_id]*pd)
#             pv.append(f["pixel_values"])
#             lp=torch.from_numpy(f["lidar_points"]).float()
#             rp=torch.from_numpy(f["radar_points"]).float()
#             nl,nr=f["n_lidar"],f["n_radar"]
#             li.append(lp[:nl] if nl>0 else lp[:1]*0)
#             ra.append(rp[:nr] if nr>0 else rp[:1]*0)
#             if self.plan and "gt_waypoints" in f:
#                 wp.append(torch.from_numpy(f["gt_waypoints"]).float())
#         r = {"input_ids":torch.tensor(ids,dtype=torch.long),
#              "labels":torch.tensor(lab,dtype=torch.long),
#              "pixel_values":torch.stack(pv),"lidar_points":li,"radar_points":ra}
#         if self.plan and wp: r["gt_waypoints"]=torch.stack(wp)
#         return r

# def parse_args():
#     p = argparse.ArgumentParser()
#     # Stage control
#     p.add_argument("--stage", type=int, default=1, choices=[1, 2],
#                    help="1=VQA only (LoRA), 2=Planning (freeze VQA, train planner)")
#     p.add_argument("--data_root", type=str, required=True)
#     p.add_argument("--output_dir", type=str, default="save/vlm_diffplan")
#     p.add_argument("--llm_model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
#     p.add_argument("--vision_model_path", type=str, default="facebook/dinov3-base")
#     p.add_argument("--batch_size", type=int, default=2)
#     p.add_argument("--gradient_accumulation_steps", type=int, default=16)
#     p.add_argument("--learning_rate", type=float, default=5e-6)
#     p.add_argument("--num_epochs", type=int, default=10)
#     p.add_argument("--warmup_steps", type=int, default=200)
#     p.add_argument("--save_steps", type=int, default=500)
#     p.add_argument("--logging_steps", type=int, default=50)
#     p.add_argument("--num_workers", type=int, default=4)
#     p.add_argument("--max_grad_norm", type=float, default=1.0)
#     p.add_argument("--fp16", action="store_true", default=True)
#     p.add_argument("--bf16", action="store_true", default=False)
#     # LoRA
#     p.add_argument("--lora_r", type=int, default=16)
#     p.add_argument("--lora_alpha", type=int, default=32)
#     p.add_argument("--lora_dropout", type=float, default=0.05)
#     # Planning
#     p.add_argument("--num_planning_tokens", type=int, default=4)
#     p.add_argument("--num_anchors", type=int, default=8)
#     p.add_argument("--planning_loss_weight", type=float, default=1.0)
#     p.add_argument("--t_trunc", type=int, default=5)
#     # Model
#     p.add_argument("--num_query_tokens", type=int, default=64)
#     p.add_argument("--qformer_layers", type=int, default=4)
#     p.add_argument("--max_lidar_points", type=int, default=40000)
#     p.add_argument("--max_radar_points", type=int, default=16000)
#     p.add_argument("--sequences", nargs="+", default=None)
#     p.add_argument("--resume_from", type=str, default=None)
#     p.add_argument("--vqa_checkpoint", type=str, default=None,
#                    help="Stage 2: path to Stage 1 VQA checkpoint")
#     return p.parse_args()

# def main():
#     args = parse_args()
#     lr = int(os.environ.get("LOCAL_RANK",-1))
#     ism = lr in (-1,0)

#     is_stage1 = (args.stage == 1)  # VQA only
#     is_stage2 = (args.stage == 2)  # Planning only
#     use_planning = is_stage2

#     if ism:
#         logger.info("="*60)
#         if is_stage1:
#             logger.info("Stage 1: VQA Training (LoRA)")
#         else:
#             logger.info("Stage 2: Planning Training (freeze VQA)")
#         logger.info("="*60)
#         logger.info(f"  LoRA: r={args.lora_r} alpha={args.lora_alpha}")
#         logger.info(f"  Planning: {'ON' if use_planning else 'OFF'}")

#     # ── Config ──
#     config = MultiModalVLMConfig(
#         llm_model_path=args.llm_model_path, vision_model_path=args.vision_model_path,
#         freeze_vision_model=True, freeze_llm_model=False,
#         use_lora=False, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
#         lora_dropout=args.lora_dropout,
#         qformer_dim=512, qformer_layers=args.qformer_layers, qformer_heads=8,
#         num_query_tokens=args.num_query_tokens,
#         lidar_output_dim=256, radar_output_dim=256,
#         image_pad_num=args.num_query_tokens,
#         # Planning: only create planner module in stage 2
#         use_planning=use_planning,
#         num_planning_tokens=args.num_planning_tokens if use_planning else 0,
#         planning_num_waypoints=8, planning_waypoint_dim=3,
#         planning_num_anchors=args.num_anchors, planning_cond_dim=256,
#         planning_denoise_hidden=256, planning_denoise_blocks=4,
#         planning_t_trunc=args.t_trunc, planning_n_infer_steps=2,
#         planning_loss_weight=args.planning_loss_weight,
#     )

#     model = MultiModalVLM(config)

#     # ── Tokenizer (load BEFORE weights to handle vocab size) ──
#     tok = AutoTokenizer.from_pretrained(config.llm_model_path)
#     proc = AutoImageProcessor.from_pretrained(config.vision_model_path)

#     # Add special tokens
#     special = ['<|image_pad|>']
#     if use_planning:
#         special.append('<|plan_pad|>')
#     na = tok.add_special_tokens({'additional_special_tokens': special})
#     if na > 0:
#         model.llm_model.resize_token_embeddings(len(tok))
#         model.tokenizer = tok
#         if ism: logger.info(f"  Added {na} tokens, vocab={len(tok)}")

#     # ── Load checkpoint ──
#     if args.vqa_checkpoint:
#         ck = Path(args.vqa_checkpoint)
#         wfs = sorted(ck.glob("*.safetensors"))+sorted(ck.glob("*.bin"))
#         sd = {}
#         for wf in wfs:
#             if ism: logger.info(f"  Loading: {wf.name}")
#             if wf.suffix==".safetensors":
#                 from safetensors.torch import load_file
#                 sd.update(load_file(str(wf)))
#             elif wf.suffix==".bin":
#                 ld=torch.load(str(wf),map_location="cpu",weights_only=False)
#                 if isinstance(ld,dict): sd.update(ld)
#         if sd:
#             # Detect checkpoint's embedding size and resize to match BEFORE loading
#             ckpt_embed_key = 'llm_model.model.embed_tokens.weight'
#             if ckpt_embed_key in sd:
#                 ckpt_vocab_size = sd[ckpt_embed_key].shape[0]
#                 cur_vocab_size = model.llm_model.get_input_embeddings().weight.shape[0]
#                 if ckpt_vocab_size != cur_vocab_size:
#                     if ism: logger.info(f"  Resizing embeddings: {cur_vocab_size} → {ckpt_vocab_size} (to match checkpoint)")
#                     model.llm_model.resize_token_embeddings(ckpt_vocab_size)

#             mi,un = model.load_state_dict(sd, strict=False)
#             if ism:
#                 pm = [k for k in mi if 'planner' in k]
#                 lm = [k for k in mi if 'lora' in k]
#                 om = [k for k in mi if 'planner' not in k and 'lora' not in k]
#                 logger.info(f"  Loaded checkpoint. Planner new: {len(pm)}, "
#                             f"LoRA new: {len(lm)}, Other missing: {len(om)}, "
#                             f"Unexpected: {len(un)}")
#                 if om: logger.warning(f"  Other missing: {om[:5]}")

#             # Now resize to the CURRENT tokenizer size (which may have <|plan_pad|> etc.)
#             cur_needed = len(tok)
#             cur_actual = model.llm_model.get_input_embeddings().weight.shape[0]
#             if cur_needed != cur_actual:
#                 if ism: logger.info(f"  Resizing embeddings: {cur_actual} → {cur_needed} (for current tokenizer)")
#                 model.llm_model.resize_token_embeddings(cur_needed)
#             model.tokenizer = tok

#     # ── Stage 2: Freeze VQA modules, only train planner ──
#     if is_stage2:
#         # Freeze everything except planner
#         for name, param in model.named_parameters():
#             if 'planner' not in name:
#                 param.requires_grad = False

#         # Count trainable
#         trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         total = sum(p.numel() for p in model.parameters())
#         if ism:
#             logger.info(f"  Stage 2: Froze VQA modules")
#             logger.info(f"  Trainable: {trainable:,} / {total:,} "
#                         f"({100*trainable/total:.1f}%)")

#     if torch.cuda.is_available():
#         model = model.cuda()
#         if ism: logger.info(f"  GPU: {torch.cuda.get_device_name()}")

#     # ── Dataset ──
#     ds = UniscpDatasetWithPlanning(
#         data_root=args.data_root, sequences=args.sequences, tokenizer=tok, processor=proc,
#         image_pad_num=config.image_pad_num, max_lidar=args.max_lidar_points,
#         max_radar=args.max_radar_points, lidar_pc_range=config.lidar_pc_range,
#         radar_pc_range=config.radar_pc_range,
#         num_planning_tokens=config.num_planning_tokens if use_planning else 0,
#         use_planning=use_planning)
#     if ism: logger.info(f"  Dataset: {len(ds)} samples")
#     if len(ds)==0: logger.error("No samples!"); return

#     # ── Initialize anchors from training data (speed × curvature grid) ──
#     if is_stage2 and hasattr(model, 'planner') and ism:
#         logger.info("  Initializing anchors from training data ...")
#         all_trajs = []
#         for s in ds.samples:
#             if s.get('waypoint_path'):
#                 wp = load_waypoints(s['waypoint_path'])
#                 if np.any(wp != 0):
#                     all_trajs.append(wp)
#         if all_trajs:
#             all_trajs_t = torch.from_numpy(np.stack(all_trajs)).float()
#             model.planner.load_anchors_from_data(
#                 all_trajs_t, save_dir=args.output_dir
#             )
#         else:
#             logger.warning("  No valid waypoints, using default anchors")

#     # ── Trainer ──
#     ta = TrainingArguments(
#         output_dir=args.output_dir, do_train=True,
#         per_device_train_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         learning_rate=args.learning_rate, num_train_epochs=args.num_epochs,
#         warmup_steps=args.warmup_steps, weight_decay=0.01, max_grad_norm=args.max_grad_norm,
#         fp16=args.fp16 and not args.bf16, bf16=args.bf16,
#         save_steps=args.save_steps, save_total_limit=3,
#         logging_steps=args.logging_steps, logging_first_step=True, report_to='tensorboard',
#         dataloader_pin_memory=False, dataloader_num_workers=args.num_workers,
#         remove_unused_columns=False, gradient_checkpointing=False,
#         lr_scheduler_type='cosine', ddp_find_unused_parameters=True)

#     trainer = Trainer(model=model, args=ta, train_dataset=ds,
#                       data_collator=Collator(tok, plan=use_planning))
#     if ism: logger.info("Starting training ...")
#     trainer.train(resume_from_checkpoint=args.resume_from or False)
#     if ism:
#         fd = os.path.join(args.output_dir,"final_model")
#         trainer.save_model(fd); trainer.save_state(); tok.save_pretrained(fd)
#         config.save_pretrained(fd)  # save config for evaluate to load
#         logger.info(f"Saved to {fd}")

# if __name__ == '__main__':
#     main()

"""
3-Stage Training for VLM + Truncated Diffusion Planning with LoRA.

Stage 1: VQA only (LoRA + QFormer + encoders)
    python train_with_planning.py --stage 1 \
        --data_root ./UNISCP --output_dir save/stage1_vqa_lora \
        --num_epochs 10 --sequences RURAL_A0 RURAL_A1 RURAL_A2 RURAL_B0 RURAL_B1 RURAL_B2

Stage 2: Planning only (freeze VQA, train planner)
    python train_with_planning.py --stage 2 \
        --data_root ./UNISCP --output_dir save/stage2_planning \
        --vqa_checkpoint save/stage1_vqa_lora/final_model \
        --num_epochs 10 --planning_loss_weight 2.0

Stage 3 (future): RL fine-tuning with GRPO
"""
import os, sys, json, argparse, logging
from pathlib import Path
from typing import Dict, List, Optional
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, TrainingArguments, Trainer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from fusionxdrive.model_planning import MultiModalVLM, MultiModalVLMConfig
from fusionxdrive.dataset import (read_pcd, get_rural_calibration, load_timestamps, find_nearest_idx,
                     SYSTEM_PROMPT, USER_PROMPT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DENSE_LABELS = ["t+0.5s","t+1.0s","t+1.5s","t+2.0s","t+2.5s","t+3.0s","t+3.5s","t+4.0s"]

def load_waypoints(path, n=8):
    """Load dense 8-point waypoints from JSON."""
    wp = np.zeros((n,3), dtype=np.float32)
    try:
        with open(path) as f: data = json.load(f)
        d = {w['label']:w for w in data.get('waypoints',[])}
        for i, lab in enumerate(DENSE_LABELS[:n]):
            if lab in d and d[lab].get('available',True):
                wp[i] = [d[lab]['x'], d[lab]['y'], d[lab].get('z',0.0)]
    except: pass
    return wp

class UniscpDatasetWithPlanning(Dataset):
    RURAL = ['RURAL_A0','RURAL_A1','RURAL_A2','RURAL_B0','RURAL_B1','RURAL_B2']
    OTHER = ['FENDUAN_1','KUNSHAN_LUCE6','NIGHT_GAOJIAOQIAO','CP_MSCLIKE','GARDEN_MSCLIKE','LOOP1_MSCLIKE']

    def __init__(self, data_root, sequences=None, tokenizer=None, processor=None,
                 image_pad_num=64, max_lidar=40000, max_radar=16000,
                 lidar_pc_range=None, radar_pc_range=None,
                 num_planning_tokens=4, use_planning=True):
        super().__init__()
        self.root = Path(data_root); self.tok = tokenizer; self.proc = processor
        self.ipad = image_pad_num; self.ml = max_lidar; self.mr = max_radar
        self.npt = num_planning_tokens; self.up = use_planning
        self.lpr = lidar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
        self.rpr = radar_pc_range or [-51.2,-51.2,-5.0,51.2,51.2,3.0]
        self.calib = get_rural_calibration()
        if sequences is None: sequences = self.RURAL + self.OTHER
        self.samples = []; nw = 0
        for sq in sequences:
            d = self.root / sq
            if d.exists(): nw += self._idx(sq, d)
        logger.info(f"Loaded {len(self.samples)} samples, {nw} with waypoints")

    def _fsd(self, sd, p):
        t = sd / p
        if not t.exists(): return None
        n = t / p
        return n if n.exists() else t

    def _idx(self, sn, sd):
        imd = self._fsd(sd,'1_IMAGE'); lid = self._fsd(sd,'2_LIDAR')
        rad = self._fsd(sd,'3_RADAR'); cd = sd/'6_CAPTION'; wd = sd/'7_PLANNING'/'WAYPOINTS'
        if not all([imd,lid,rad]) or not cd.exists(): return 0
        tfs = [imd/'timestamp_image_left.txt', lid/'timestamp_lidar.txt', rad/'timestamp_radar.txt']
        if not all(f.exists() for f in tfs): return 0
        tsi = load_timestamps(str(tfs[0]))
        tsl = load_timestamps(str(tfs[1]))
        tsr = load_timestamps(str(tfs[2]))
        ld = imd/'LEFT'; lpd = lid/'PCD'; rpd = rad/'PCD'
        if not all(d.exists() for d in [ld,lpd,rpd]): return 0
        c = 0
        for ix, ts in tsi:
            ip = ld/f"{ix}.png"; cp = cd/f"{ix}.json"
            if not ip.exists() or not cp.exists(): continue
            li = find_nearest_idx(ts, tsl, max_diff=0.15)
            ri = find_nearest_idx(ts, tsr, max_diff=0.15)
            if li is None or ri is None: continue
            lp = lpd/f"{li}.pcd"; rp = rpd/f"{ri}.pcd"
            if not lp.exists() or not rp.exists(): continue
            wp = wd/f"{ix}.json"; hw = wp.exists()
            if hw: c += 1
            self.samples.append({'seq':sn,'img_idx':ix,'img_path':str(ip),
                'lidar_path':str(lp),'radar_path':str(rp),'caption_path':str(cp),
                'waypoint_path':str(wp) if hw else None})
        return c

    def __len__(self): return len(self.samples)

    def _ll(self, p):
        try:
            d = read_pcd(p, ['x','y','z','intensity'])
            x,y,z = d.get('x',np.zeros(0)),d.get('y',np.zeros(0)),d.get('z',np.zeros(0))
            i = d.get('intensity',np.zeros_like(x))
            if len(x)==0: return np.zeros((0,4),dtype=np.float32)
            pts = np.stack([x,y,z,i],-1).astype(np.float32)
            pts = pts[np.all(np.isfinite(pts),-1)]; pts = pts[np.any(pts[:,:3]!=0,-1)]
            return pts
        except: return np.zeros((0,4),dtype=np.float32)

    def _lr(self, p):
        try:
            d = read_pcd(p, ['x','y','z','doppler','power','recoveredSpeed'])
            x,y,z = d.get('x',np.zeros(0)),d.get('y',np.zeros(0)),d.get('z',np.zeros(0))
            dp,pw,sp = d.get('doppler',np.zeros_like(x)),d.get('power',np.zeros_like(x)),d.get('recoveredSpeed',np.zeros_like(x))
            if len(x)==0: return np.zeros((0,6),dtype=np.float32)
            pts = np.stack([x,y,z,dp,pw,sp],-1).astype(np.float32)
            return pts[np.all(np.isfinite(pts),-1)]
        except: return np.zeros((0,6),dtype=np.float32)

    @staticmethod
    def _sub(pts, mx):
        if len(pts)==0: return np.zeros((1,pts.shape[1] if pts.ndim>1 else 4),dtype=np.float32)
        return pts[np.random.choice(len(pts),mx,replace=False)] if len(pts)>mx else pts

    def __getitem__(self, i):
        s = self.samples[i]
        try: img = Image.open(s['img_path']).convert('RGB')
        except: img = Image.new('RGB',(720,540),'black')
        lidar = self._ll(s['lidar_path'])
        if len(lidar)>0: lidar = self.calib.crop_lidar_to_fov(lidar)
        lidar = self._sub(lidar, self.ml)
        radar = self._lr(s['radar_path'])
        if len(radar)>0: radar = self.calib.crop_radar_to_fov(radar)
        radar = self._sub(radar, self.mr)
        try:
            with open(s['caption_path']) as f: cap = json.load(f)
        except:
            cap = {"weather":{"condition":"unknown","illumination":"unknown"},
                   "traffic_light":{"present":"unknown","state":"unknown"},
                   "traffic_sign":{"present":"unknown","category":"unknown"},
                   "participants":{"count":"0","objects":[]},
                   "hazard_region":{"present":"unknown","type":"unknown","direction":"unknown"},
                   "forward_drivability":{"status":"unknown"},
                   "lane_keeping":{"status":"unknown","deviation":"unknown"},
                   "driving_advice":{"action":"unknown"},"explanation":{"reason":"unknown"}}

        gwp = np.zeros((8,3),dtype=np.float32)
        if self.up and s.get('waypoint_path'):
            gwp = load_waypoints(s['waypoint_path'])

        pv = self.proc(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        ans = json.dumps(cap, ensure_ascii=False)
        msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":"<image>\n"+USER_PROMPT}]
        qt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        qt = qt.replace('<image>', '<|image_pad|>'*self.ipad)
        qi = self.tok(qt)["input_ids"]
        ai = self.tok(ans + self.tok.eos_token)["input_ids"]

        # Only append plan tokens if planning is enabled
        if self.up and self.npt > 0:
            pi = self.tok('<|plan_pad|>'*(self.npt+1))["input_ids"]
        else:
            pi = []

        all_ids = qi + ai + pi
        all_lab = [self.tok.pad_token_id]*len(qi) + ai + [self.tok.pad_token_id]*len(pi)

        result = {"input_ids":all_ids[:-1], "labels":all_lab[1:], "pixel_values":pv,
                  "lidar_points":lidar.astype(np.float32), "radar_points":radar.astype(np.float32),
                  "n_lidar":len(lidar), "n_radar":len(radar)}

        if self.up:
            result["gt_waypoints"] = gwp

        return result

class Collator:
    def __init__(self, tok, plan=True): self.tok=tok; self.plan=plan
    def __call__(self, fs):
        mx = max(len(f["input_ids"]) for f in fs)
        ids,lab,pv,li,ra,wp = [],[],[],[],[],[]
        for f in fs:
            pd = mx-len(f["input_ids"])
            ids.append(f["input_ids"]+[self.tok.pad_token_id]*pd)
            lab.append(f["labels"]+[self.tok.pad_token_id]*pd)
            pv.append(f["pixel_values"])
            lp=torch.from_numpy(f["lidar_points"]).float()
            rp=torch.from_numpy(f["radar_points"]).float()
            nl,nr=f["n_lidar"],f["n_radar"]
            li.append(lp[:nl] if nl>0 else lp[:1]*0)
            ra.append(rp[:nr] if nr>0 else rp[:1]*0)
            if self.plan and "gt_waypoints" in f:
                wp.append(torch.from_numpy(f["gt_waypoints"]).float())
        r = {"input_ids":torch.tensor(ids,dtype=torch.long),
             "labels":torch.tensor(lab,dtype=torch.long),
             "pixel_values":torch.stack(pv),"lidar_points":li,"radar_points":ra}
        if self.plan and wp: r["gt_waypoints"]=torch.stack(wp)
        return r

def parse_args():
    p = argparse.ArgumentParser()
    # Stage control
    p.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="1=VQA only (LoRA), 2=Planning (freeze VQA, train planner)")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="save/vlm_diffplan")
    p.add_argument("--llm_model_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--vision_model_path", type=str, default="facebook/dinov3-base")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=False)
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # Planning
    p.add_argument("--num_planning_tokens", type=int, default=4)
    p.add_argument("--num_anchors", type=int, default=8)
    p.add_argument("--planning_loss_weight", type=float, default=1.0)
    p.add_argument("--t_trunc", type=int, default=5)
    p.add_argument("--cond_dim", type=int, default=512,
                   help="Planner condition dimension")
    p.add_argument("--denoise_hidden", type=int, default=512,
                   help="Denoiser hidden dimension")
    p.add_argument("--denoise_blocks", type=int, default=6,
                   help="Number of denoiser FiLM blocks")
    # Model
    p.add_argument("--num_query_tokens", type=int, default=64)
    p.add_argument("--qformer_layers", type=int, default=4)
    p.add_argument("--max_lidar_points", type=int, default=40000)
    p.add_argument("--max_radar_points", type=int, default=16000)
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--vqa_checkpoint", type=str, default=None,
                   help="Stage 2: path to Stage 1 VQA checkpoint")
    return p.parse_args()

def main():
    args = parse_args()
    lr = int(os.environ.get("LOCAL_RANK",-1))
    ism = lr in (-1,0)

    is_stage1 = (args.stage == 1)  # VQA only
    is_stage2 = (args.stage == 2)  # Planning only
    use_planning = is_stage2

    if ism:
        logger.info("="*60)
        if is_stage1:
            logger.info("Stage 1: VQA Training (LoRA)")
        else:
            logger.info("Stage 2: Planning Training (freeze VQA)")
        logger.info("="*60)
        logger.info(f"  LoRA: r={args.lora_r} alpha={args.lora_alpha}")
        logger.info(f"  Planning: {'ON' if use_planning else 'OFF'}")

    # ── Config ──
    config = MultiModalVLMConfig(
        llm_model_path=args.llm_model_path, vision_model_path=args.vision_model_path,
        freeze_vision_model=True, freeze_llm_model=False,
        use_lora=False, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        qformer_dim=512, qformer_layers=args.qformer_layers, qformer_heads=8,
        num_query_tokens=args.num_query_tokens,
        lidar_output_dim=256, radar_output_dim=256,
        image_pad_num=args.num_query_tokens,
        # Planning: only create planner module in stage 2
        use_planning=use_planning,
        num_planning_tokens=args.num_planning_tokens if use_planning else 0,
        planning_num_waypoints=8, planning_waypoint_dim=3,
        planning_num_anchors=args.num_anchors, planning_cond_dim=args.cond_dim,
        planning_denoise_hidden=args.denoise_hidden, planning_denoise_blocks=args.denoise_blocks,
        planning_t_trunc=args.t_trunc, planning_n_infer_steps=2,
        planning_loss_weight=args.planning_loss_weight,
    )

    model = MultiModalVLM(config)

    # ── Tokenizer (load BEFORE weights to handle vocab size) ──
    tok = AutoTokenizer.from_pretrained(config.llm_model_path)
    proc = AutoImageProcessor.from_pretrained(config.vision_model_path)

    # Add special tokens
    special = ['<|image_pad|>']
    if use_planning:
        special.append('<|plan_pad|>')
    na = tok.add_special_tokens({'additional_special_tokens': special})
    if na > 0:
        model.llm_model.resize_token_embeddings(len(tok))
        model.tokenizer = tok
        if ism: logger.info(f"  Added {na} tokens, vocab={len(tok)}")

    # ── Load checkpoint ──
    if args.vqa_checkpoint:
        ck = Path(args.vqa_checkpoint)
        wfs = sorted(ck.glob("*.safetensors"))+sorted(ck.glob("*.bin"))
        sd = {}
        for wf in wfs:
            if ism: logger.info(f"  Loading: {wf.name}")
            if wf.suffix==".safetensors":
                from safetensors.torch import load_file
                sd.update(load_file(str(wf)))
            elif wf.suffix==".bin":
                ld=torch.load(str(wf),map_location="cpu",weights_only=False)
                if isinstance(ld,dict): sd.update(ld)
        if sd:
            # Detect checkpoint's embedding size and resize to match BEFORE loading
            ckpt_embed_key = 'llm_model.model.embed_tokens.weight'
            if ckpt_embed_key in sd:
                ckpt_vocab_size = sd[ckpt_embed_key].shape[0]
                cur_vocab_size = model.llm_model.get_input_embeddings().weight.shape[0]
                if ckpt_vocab_size != cur_vocab_size:
                    if ism: logger.info(f"  Resizing embeddings: {cur_vocab_size} → {ckpt_vocab_size} (to match checkpoint)")
                    model.llm_model.resize_token_embeddings(ckpt_vocab_size)

            mi,un = model.load_state_dict(sd, strict=False)
            if ism:
                pm = [k for k in mi if 'planner' in k]
                lm = [k for k in mi if 'lora' in k]
                om = [k for k in mi if 'planner' not in k and 'lora' not in k]
                logger.info(f"  Loaded checkpoint. Planner new: {len(pm)}, "
                            f"LoRA new: {len(lm)}, Other missing: {len(om)}, "
                            f"Unexpected: {len(un)}")
                if om: logger.warning(f"  Other missing: {om[:5]}")

            # Now resize to the CURRENT tokenizer size (which may have <|plan_pad|> etc.)
            cur_needed = len(tok)
            cur_actual = model.llm_model.get_input_embeddings().weight.shape[0]
            if cur_needed != cur_actual:
                if ism: logger.info(f"  Resizing embeddings: {cur_actual} → {cur_needed} (for current tokenizer)")
                model.llm_model.resize_token_embeddings(cur_needed)
            model.tokenizer = tok

    # ── Stage 2: Freeze VQA modules, only train planner ──
    if is_stage2:
        # Freeze everything except planner
        for name, param in model.named_parameters():
            if 'planner' not in name:
                param.requires_grad = False

        # Count trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if ism:
            logger.info(f"  Stage 2: Froze VQA modules")
            logger.info(f"  Trainable: {trainable:,} / {total:,} "
                        f"({100*trainable/total:.1f}%)")

    if torch.cuda.is_available():
        model = model.cuda()
        if ism: logger.info(f"  GPU: {torch.cuda.get_device_name()}")

    # ── Dataset ──
    ds = UniscpDatasetWithPlanning(
        data_root=args.data_root, sequences=args.sequences, tokenizer=tok, processor=proc,
        image_pad_num=config.image_pad_num, max_lidar=args.max_lidar_points,
        max_radar=args.max_radar_points, lidar_pc_range=config.lidar_pc_range,
        radar_pc_range=config.radar_pc_range,
        num_planning_tokens=config.num_planning_tokens if use_planning else 0,
        use_planning=use_planning)
    if ism: logger.info(f"  Dataset: {len(ds)} samples")
    if len(ds)==0: logger.error("No samples!"); return

    # ── Initialize anchors from training data (speed × curvature grid) ──
    if is_stage2 and hasattr(model, 'planner') and ism:
        logger.info("  Initializing anchors from training data ...")
        all_trajs = []
        for s in ds.samples:
            if s.get('waypoint_path'):
                wp = load_waypoints(s['waypoint_path'])
                if np.any(wp != 0):
                    all_trajs.append(wp)
        if all_trajs:
            all_trajs_t = torch.from_numpy(np.stack(all_trajs)).float()
            model.planner.load_anchors_from_data(
                all_trajs_t, save_dir=args.output_dir
            )
        else:
            logger.warning("  No valid waypoints, using default anchors")

    # ── Trainer ──
    ta = TrainingArguments(
        output_dir=args.output_dir, do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps, weight_decay=0.01, max_grad_norm=args.max_grad_norm,
        fp16=args.fp16 and not args.bf16, bf16=args.bf16,
        save_steps=args.save_steps, save_total_limit=3,
        logging_steps=args.logging_steps, logging_first_step=True, report_to='tensorboard',
        dataloader_pin_memory=False, dataloader_num_workers=args.num_workers,
        remove_unused_columns=False, gradient_checkpointing=False,
        lr_scheduler_type='cosine', ddp_find_unused_parameters=True)

    trainer = Trainer(model=model, args=ta, train_dataset=ds,
                      data_collator=Collator(tok, plan=use_planning))
    if ism: logger.info("Starting training ...")
    trainer.train(resume_from_checkpoint=args.resume_from or False)
    if ism:
        fd = os.path.join(args.output_dir,"final_model")
        trainer.save_model(fd); trainer.save_state(); tok.save_pretrained(fd)
        config.save_pretrained(fd)  # save config for evaluate to load
        logger.info(f"Saved to {fd}")

if __name__ == '__main__':
    main()