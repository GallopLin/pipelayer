#!/usr/bin/env python3
"""2 卡 PP stage checkpoint 的恢复侧布局验证（无需构建 PP stage 模型）。

覆盖：resolve_multistream_checkpoint 的 stage_{rank} 布局分支、v3 metadata
解析、extra blob 直读（crc 校验）。模型级逐位对账由 tests/
test_distributed_resume.py 覆盖（同一 loader 代码路径）。
"""
import argparse
import json
import os
import sys

p = argparse.ArgumentParser()
p.add_argument("--moddir", default="/root/code/pccheck/checkpoint_eval/pccheck")
p.add_argument("--ckpt-dir", default="/root/ms_checkpoints_bench")
p.add_argument("--pipelayer", default="/root/download/pipelayer/src")
p.add_argument("--ranks", type=int, default=2)
args = p.parse_args()

sys.path.insert(0, args.moddir)
sys.path.insert(0, args.pipelayer)

import numpy as np
from pipelayer.distributed import resolve_multistream_checkpoint
from multistream_checkpoint import MultiStreamCheckpoint

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    ok_all &= bool(ok)
    return bool(ok)


for rank in range(args.ranks):
    print(f"--- rank {rank} ---", flush=True)
    r = resolve_multistream_checkpoint(args.ckpt_dir, rank=rank)
    chk = os.path.join(r.chkpt_dir, r.checkpoint_file)
    meta_path = os.path.join(r.chkpt_dir, r.metadata_file)
    check("stage layout resolved", r.chkpt_dir.endswith(f"stage_{rank}")
          and os.path.exists(chk) and os.path.exists(meta_path),
          f"(slot={r.parall_iter}, steps={r.training_state.get('completed_steps')})")

    meta = json.load(open(meta_path))
    check("metadata v3 + master-empty(3N) + extra enabled",
          meta.get("version") == 3
          and meta["stream_sizes"][1] == 0
          and meta.get("extra_state", {}).get("enabled"),
          f"(streams={[s // 1000000 for s in meta['stream_sizes']]}M floats)")

    # extra blob 直读：按 v3 布局定位该槽位 extra 流并 crc 校验解析
    stride = int(meta["checkpoint_stride_floats"])
    extra_off = sum(meta["stream_sizes"][:4])
    extra_floats = int(meta["extra_state"]["floats"])
    slot = int(r.parall_iter if r.parall_iter is not None
               else meta.get("latest_parall_iter", 0))
    base = (int(meta["max_async"]) + 4) * 4096
    byte_pos = (base + slot * stride + extra_off) * 4
    raw = np.memmap(chk, dtype=np.uint8, mode="r",
                    offset=byte_pos, shape=(extra_floats * 4,))
    try:
        blob, _plen = MultiStreamCheckpoint.parse_extra_state_blob(bytes(raw))
        check("extra blob crc + parse", True,
              f"(optimizer_step={blob.get('optimizer_step')}, "
              f"parall_iter={blob.get('parall_iter')})")
        check("blob slot consistency",
              int(blob.get("parall_iter", -1)) == slot, f"(slot={slot})")
    except ValueError as e:
        check("extra blob crc + parse", False, f"({e})")

print(f"\n[stage-resolve] {'ALL PASS' if ok_all else 'FAILURES PRESENT'}", flush=True)
os._exit(0 if ok_all else 1)
