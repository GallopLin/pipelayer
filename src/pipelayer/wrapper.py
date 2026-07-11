from __future__ import annotations

import atexit
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import torch
import torch.nn as nn

from .checkpointing import PipelinedStateLoader


class PipelayerModelWrapper(nn.Module):
    """
    在加载期间通过 forward pre-hook 实现按需同步的模型包装器：
    - 后台线程按 chunk 顺序加载检查点数据到 GPU。
    - 在原始模型的子模块上注册 pre-hook，每个模块执行前等待其所需 chunk 就绪。
    - 原始模型的 forward 逻辑（残差连接、注意力等）完全保留。
    - 全部加载完成后自动移除 hook，零额外开销。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chkpt_dir: str | None = None,
        load_checkpoint: bool = False,
        device: str | None = None,
        loader_cls: type[PipelinedStateLoader] = PipelinedStateLoader,
        loader_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.original_model = model.to(self.device)
        self.optimizer = optimizer
        self.loader: PipelinedStateLoader | None = None
        self.chkpt_dir = chkpt_dir
        self.loader_cls = loader_cls
        self.loader_kwargs = loader_kwargs or {}

        # 状态管理
        self._loading_complete = threading.Event()
        self._wait_hooks: List[torch.utils.hooks.RemovableHook] = []

        # pipelining 验证日志
        self._init_time = time.time()
        self._chunk_loaded_times: Dict[int, float] = {}   # chunk_id -> 相对 init 的加载完成时间
        self._hook_wait_log: List[Tuple[int, float, float]] = []  # (chunk_id, wait_start, blocked_duration)
        self._first_forward_time: float = 0.0
        self._pipelining_verified = False
        self._extra_state_cache: Dict[str, Any] | None = None

        # 注册清理函数
        atexit.register(self.cleanup)
        self._cleaned = False

        # 只有当需要加载检查点时才初始化加载器并启动加载
        if load_checkpoint:
            if not chkpt_dir:
                raise ValueError("chkpt_dir must be provided when load_checkpoint is True")
            self._initialize_loader()
            self._setup_chunk_wait_hooks()
            self._start_loading_monitor()

    # ------------------------------------------------------------------
    # Loader 初始化
    # ------------------------------------------------------------------

    def _initialize_loader(self) -> None:
        """初始化加载器"""
        self.loader = self.loader_cls(
            self.original_model,
            self.optimizer,
            self.chkpt_dir,
            device=str(self.device),
            **self.loader_kwargs,
        )

    # ------------------------------------------------------------------
    # Hook 机制：在原始模型子模块上注册 pre-hook 实现按需等待
    # ------------------------------------------------------------------

    def _setup_chunk_wait_hooks(self) -> None:
        """
        为拥有检查点参数的每个子模块注册 forward pre-hook。
        hook 在模块执行前调用 wait_for_chunk，确保该模块所需的
        所有 chunk 数据已从后台线程拷贝到 GPU。

        这样原始模型的 forward 逻辑完全不变（残差连接、注意力等），
        只是在每个子模块入口处插入了同步点。
        """
        assert self.loader is not None

        # 1) 建立 module_name -> 所需 chunk_ids 的映射
        # split 分组下一个参数可横跨多个 chunk：必须等待全部覆盖 chunk（正确性
        # 要求，非优化）。param_name_to_chunks 由 MultiStreamStateLoader 提供，
        # 旧 loader 回退到单值映射。
        name_to_chunks = getattr(self.loader, "param_name_to_chunks", None)
        if not name_to_chunks:
            name_to_chunks = {k: {v} for k, v in self.loader.param_name_to_chunk.items()}
        module_to_chunks: Dict[str, Set[int]] = defaultdict(set)
        for pname, chunk_ids in name_to_chunks.items():
            # pname 形如 "model.decoder.layers.0.self_attn.k_proj.weight"
            # 取最后一个 "." 之前的部分作为 module_name
            dot_pos = pname.rfind(".")
            if dot_pos > 0:
                module_name = pname[:dot_pos]
                module_to_chunks[module_name].update(chunk_ids)

        # 2) 在对应模块上注册 pre-hook
        all_modules = dict(self.original_model.named_modules())
        for module_name, chunk_ids in module_to_chunks.items():
            module = all_modules.get(module_name)
            if module is None:
                continue
            sorted_ids = sorted(chunk_ids)
            hook = module.register_forward_pre_hook(self._make_wait_hook(sorted_ids))
            self._wait_hooks.append(hook)

        print(f"[PipeLayer] Registered {len(self._wait_hooks)} chunk-wait hooks "
              f"across {self.loader.num_chunks} chunks")

    def _make_wait_hook(self, chunk_ids: List[int]):
        """创建一个 pre-hook 闭包，在模块执行前等待指定 chunk 就绪。"""
        def hook(module: nn.Module, input: Any) -> None:
            if self._first_forward_time == 0.0:
                self._first_forward_time = time.time()
            loader = self.loader
            if loader is not None:
                for cid in chunk_ids:
                    t0 = time.time()
                    already_loaded = loader.is_chunk_loaded(cid)
                    loader.wait_for_chunk(cid)
                    elapsed = time.time() - t0
                    # 只记录首次等待
                    if cid not in self._chunk_loaded_times:
                        self._chunk_loaded_times[cid] = time.time() - self._init_time
                    if elapsed > 0.001 and not already_loaded:
                        self._hook_wait_log.append((cid, t0 - self._init_time, elapsed))
        return hook

    def _remove_chunk_wait_hooks(self) -> None:
        """移除所有 chunk 等待 hook。"""
        for hook in self._wait_hooks:
            hook.remove()
        self._wait_hooks.clear()

    # ------------------------------------------------------------------
    # 后台监控：检测加载完成后清理 hook 和 loader
    # ------------------------------------------------------------------

    def _start_loading_monitor(self) -> None:
        """启动后台监控线程，检测加载完成并输出 pipelining 验证报告"""

        def monitor_loading() -> None:
            try:
                chunk_load_times: Dict[int, float] = {}
                while self.loader and not self._loading_complete.is_set():
                    all_loaded = True
                    for i in range(self.loader.num_chunks):
                        if i not in chunk_load_times and self._check_chunk_loaded(i):
                            chunk_load_times[i] = time.time() - self._init_time
                        if not self._check_chunk_loaded(i):
                            all_loaded = False

                    if all_loaded:
                        self._print_pipelining_report(chunk_load_times)
                        self._finish_loading()
                        break

                    time.sleep(0.01)
            except Exception as e:
                print(f"[PipeLayer] Loading monitor error: {e}")

        monitor_thread = threading.Thread(target=monitor_loading, daemon=True)
        monitor_thread.start()

    def _check_chunk_loaded(self, chunk_idx: int) -> bool:
        """非阻塞检查块是否已【完整】加载（含 optimizer 状态流）。

        weights-first 下 param 先就绪不算完成——若按 param 判据，监控线程会
        提前 stop loader，杀掉第二阶段（moments/master）的恢复。
        """
        if self.loader is None:
            return False
        if hasattr(self.loader, "is_chunk_fully_loaded"):
            return self.loader.is_chunk_fully_loaded(chunk_idx)
        return self.loader.is_chunk_loaded(chunk_idx)

    def _print_pipelining_report(self, chunk_load_times: Dict[int, float]) -> None:
        """输出 pipelining 验证报告"""
        print("\n" + "=" * 60)
        print("[PipeLayer] Pipelining Verification Report")
        print("=" * 60)

        # chunk 加载时间线
        print(f"  Init time (T=0): wrapper.__init__ start")
        for cid in sorted(chunk_load_times):
            print(f"  Chunk {cid} loaded at T+{chunk_load_times[cid]:.3f}s")

        first_fwd = self._first_forward_time - self._init_time if self._first_forward_time > 0 else None
        if first_fwd is not None:
            print(f"  First forward() at T+{first_fwd:.3f}s")
        else:
            print(f"  First forward() not yet called")

        # 判断是否有真正的 pipelining
        all_loaded_time = max(chunk_load_times.values()) if chunk_load_times else 0
        if first_fwd is not None and first_fwd < all_loaded_time:
            print(f"\n  [OK] PIPELINING DETECTED: forward started {all_loaded_time - first_fwd:.3f}s "
                  f"before all chunks loaded")
            # 哪些 chunk 在 forward 开始后才加载完
            late_chunks = {c: t for c, t in chunk_load_times.items() if t > first_fwd}
            if late_chunks:
                print(f"  Chunks loaded during forward: {sorted(late_chunks.keys())}")
        else:
            print(f"\n  [WARN] NO PIPELINING: all chunks loaded before first forward()")
            if first_fwd is not None:
                print(f"  All loaded at T+{all_loaded_time:.3f}s, forward at T+{first_fwd:.3f}s "
                      f"(gap: {first_fwd - all_loaded_time:.3f}s)")

        # hook 阻塞记录
        if self._hook_wait_log:
            print(f"\n  Hook blocking events ({len(self._hook_wait_log)} times):")
            for cid, wait_start, duration in self._hook_wait_log[:10]:
                print(f"    Chunk {cid}: waited {duration:.3f}s at T+{wait_start:.3f}s")
        else:
            print(f"\n  No hook blocking events (all chunks ready before forward)")

        # weights-first：param-ready vs full-ready 双时间线（恢复侧重叠证据）
        loader = self.loader
        pt = getattr(loader, "_param_ready_time", None)
        ft = getattr(loader, "_full_ready_time", None)
        if getattr(loader, "weights_first", False) and pt:
            print(f"\n  [weights-first] param-ready vs full-ready (s since loader init):")
            for cid in sorted(pt):
                fr = (ft or {}).get(cid)
                full_desc = f"full T+{fr:.3f}" if fr is not None else "full pending"
                print(f"    Chunk {cid}: param T+{pt[cid]:.3f}  {full_desc}")

        print("=" * 60 + "\n")

    def _finish_loading(self) -> None:
        """加载完成：移除 hook、释放 loader 资源。

        释放前缓存 extra state（非分层状态 blob）——loader 释放后 reader 句柄
        关闭，finish_optimizer_state 若晚于监控线程到达仍能拿到状态。
        """
        self._loading_complete.set()
        self._remove_chunk_wait_hooks()

        if self.loader:
            if self._extra_state_cache is None and hasattr(self.loader, "load_extra_state"):
                try:
                    self._extra_state_cache = self.loader.load_extra_state()
                except Exception as e:
                    print(f"[PipeLayer] extra-state read failed during finish: {e}")
            self.loader.stop()
            self.loader = None

        import gc
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def finish_optimizer_state(self, scheduler=None, scaler=None, set_rng: bool = True):
        """首次 optimizer.step() 前调用：等待全部状态流（moments/master）恢复
        完成，并应用非分层状态（RNG/LR scheduler/grad scaler）。

        Returns: extra blob 的 user dict（如 dataloader 位置），无则 None。
        """
        loader = self.loader
        extra = None
        if loader is not None:
            if hasattr(loader, "wait_all_full"):
                loader.wait_all_full()
            if hasattr(loader, "load_extra_state"):
                extra = loader.load_extra_state()
        if extra is None:
            extra = self._extra_state_cache
        if extra is None:
            return None
        self._extra_state_cache = extra
        from .checkpointing import MultiStreamStateLoader
        return MultiStreamStateLoader.apply_extra_state(
            extra, scheduler=scheduler, scaler=scaler, set_rng=set_rng)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        始终调用原始模型的 forward。
        - 加载期间：子模块上的 pre-hook 自动在每个模块执行前等待所需 chunk。
        - 加载完成后：hook 已移除，零额外开销。
        """
        return self.original_model(*args, **kwargs)

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """显式清理资源"""
        if not getattr(self, "_cleaned", False):
            self._loading_complete.set()
            self._remove_chunk_wait_hooks()
            if hasattr(self, "loader") and self.loader is not None:
                self.loader.stop()
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            self._cleaned = True

    def __del__(self) -> None:
        if not getattr(self, "_cleaned", False):
            try:
                self.cleanup()
            except Exception:
                pass
