#!/usr/bin/env python3
"""distributed.py 端到端测试（单进程模拟 rank0）。

覆盖合并版多卡恢复全链路：
  resolve_multistream_checkpoint（single-process 布局发现）
  → MultiStreamStateLoader（v3 格式）
  → ForwardLoadGate（param_name_to_chunks 集合语义 pre-hook）
  → forward 触发按需等待 → params 逐位对账
  → optimizer step guard（wait_update_ready）→ moments 逐位对账
  → get_restore_stats / load_extra_state / cleanup

前置：bench_load_roundtrip.py --phase save --config fp32_split 已产出
  <ckpt-dir>/rt_fp32_split.chk{,.metadata.json,.truth.pt}
"""
import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--moddir", default="/root/code/pccheck/checkpoint_eval/pccheck")
p.add_argument("--ckpt-dir", default="/data/linzhicheng_bench")
p.add_argument("--pipelayer", default="/root/download/pipelayer/src")
args = p.parse_args()

sys.path.insert(0, args.moddir)
sys.path.insert(0, args.pipelayer)

import torch
import torch.nn as nn
import torch.nn.functional as F

from pipelayer.distributed import (
    resolve_multistream_checkpoint,
    start_distributed_multistream_resume,
)

V, H, L, FF = 32000, 1024, 4, 4096
dev = "cuda:0"


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H)
        self.q, self.k, self.v, self.o = (nn.Linear(H, H) for _ in range(4))
        self.ln2 = nn.LayerNorm(H)
        self.up, self.down = nn.Linear(H, FF), nn.Linear(FF, H)

    def forward(self, x):
        h = self.ln1(x)
        att = torch.softmax(self.q(h) @ self.k(h).transpose(-1, -2) / H ** 0.5, dim=-1)
        x = x + self.o(att @ self.v(h))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.lnf = nn.LayerNorm(H)
        self.head = nn.Linear(H, V, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


ok_all = True


def check(name, ok, detail=""):
    global ok_all
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name} {detail}", flush=True)
    ok_all &= bool(ok)
    return bool(ok)


CKPT = os.path.join(args.ckpt_dir, "rt_fp32_split.chk")
truth = torch.load(CKPT + ".truth.pt", map_location="cpu")

# 1. 布置 single-process 发现布局（multistream_info.json + training_state.json）
info = {
    "lib_path": os.path.join(args.moddir, "libtest_ssd.so"),
    "metadata_file": os.path.basename(CKPT) + ".metadata.json",
    "checkpoint_file": os.path.basename(CKPT),
}
with open(os.path.join(args.ckpt_dir, "multistream_info.json"), "w") as f:
    json.dump(info, f)
with open(os.path.join(args.ckpt_dir, "training_state.json"), "w") as f:
    json.dump({"completed_steps": int(truth["iteration"]),
               "slot": int(truth["final_slot"]), "rank": 0}, f)

resolved = resolve_multistream_checkpoint(args.ckpt_dir, rank=0)
check("resolve single-process layout",
      resolved.checkpoint_file == os.path.basename(CKPT)
      and resolved.parall_iter is None,
      "(slot chosen by loader from the published record)")

# 2. 零初始化模型（证明数据来自 checkpoint 而非初始化）+ 预建 optimizer state
torch.manual_seed(12345)
model = GPT().to(dev)
with torch.no_grad():
    for prm in model.parameters():
        prm.zero_()
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. 启动分布式恢复（engine=None → guard 挂在 optimizer.step 上）
handle = start_distributed_multistream_resume(
    model=model,
    optimizer=opt,
    checkpoint_dir=args.ckpt_dir,
    rank=0,
    device=dev,
    engine=None,
    install_forward_hooks=True,
    guard_optimizer_step=True,
)
check("handle created + gate installed",
      handle.gate is not None and handle.gate._hook_module_count > 0,
      f"(hook modules={handle.gate._hook_module_count})")

# 4. forward：pre-hook 触发按需等待（ForwardLoadGate → wait_for_forward_group）
ids = torch.randint(0, V, (2, 64), device=dev)
with torch.no_grad():
    _ = model(ids)
check("forward completed under gate", True)

# 5. params 逐位对账
n_bad = sum(1 for n, prm in model.named_parameters()
            if not torch.equal(prm.detach().float().cpu(), truth["params"][n]))
check("params bit-exact after forward", n_bad == 0, f"(mismatch={n_bad})")

# 6. optimizer.step guard：step 前应自动 wait_update_ready
# 全参数零梯度探针：每个参数都得到 grad=0（Adam 对 grad=None 的参数会跳过
# 且不衰减 state，会破坏第 7 步的 β 还原对账）
loss_probe = sum(prm.sum() for prm in model.parameters()) * 0
loss_probe.backward()
opt.step()  # guard 在此触发 wait_update_ready 并自动卸载
check("optimizer step guard fired + update ready",
      handle.is_complete() and handle.loader.is_update_ready())

# 7. moments 逐位对账（zero-grad step 后 exp_avg 衰减 β1、exp_avg_sq 衰减 β2 —— 用还原比对）
b1, b2 = 0.9, 0.999
n_bad_m = 0
for n, prm in model.named_parameters():
    st = opt.state[prm]
    ea = (st["exp_avg"].detach().float().view(-1).cpu() / b1)
    eas = (st["exp_avg_sq"].detach().float().view(-1).cpu() / b2)
    if not torch.allclose(ea, truth["exp_avg"][n], rtol=1e-6, atol=1e-12):
        n_bad_m += 1
    if not torch.allclose(eas, truth["exp_avg_sq"][n], rtol=1e-6, atol=1e-12):
        n_bad_m += 1
check("moments restored (β-decay reversed)", n_bad_m == 0, f"(mismatch={n_bad_m})")

# 8. stats + extra state + training_state 透传
stats = handle.get_restore_stats()
check("restore stats", stats.get("num_chunks", 0) > 0
      and stats["full_chunks_loaded"] == stats["num_chunks"], f"({stats})")
check("loader picked the published slot",
      handle.loader.parall_iter == int(truth["final_slot"])
      and handle.loader.slot_source == "published",
      f"(slot={handle.loader.parall_iter}, source={handle.loader.slot_source})")
extra = handle.loader.load_extra_state()
check("extra state blob readable", extra is not None
      and int(extra.get("optimizer_step", -1)) == int(truth["step"]),
      f"(step={extra.get('optimizer_step')} vs truth {truth['step']})")
check("training_state passthrough",
      handle.training_state.get("completed_steps") == int(truth["iteration"]))

handle.cleanup()
check("cleanup", True)

print(f"\n[distributed-resume] {'ALL PASS' if ok_all else 'FAILURES PRESENT'}", flush=True)
os._exit(0 if ok_all else 1)
