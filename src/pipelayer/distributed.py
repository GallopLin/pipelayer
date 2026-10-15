from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from .checkpointing import MultiStreamStateLoader


@dataclass
class ResolvedMultiStreamCheckpoint:
    chkpt_dir: str
    metadata_file: str
    checkpoint_file: str
    lib_path: str
    rank: int
    training_state: Dict[str, Any]
    parall_iter: Optional[int] = None


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_rank(rank: Optional[int] = None) -> int:
    if rank is not None:
        return int(rank)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return torch.device(f"cuda:{int(local_rank)}")
    return torch.device(f"cuda:{torch.cuda.current_device()}")


def _latest_iteration(checkpoint_dir: str) -> Optional[int]:
    latest = None
    if not os.path.isdir(checkpoint_dir):
        return None
    for name in os.listdir(checkpoint_dir):
        match = re.fullmatch(r"iter_(\d+)", name)
        if match is None:
            continue
        value = int(match.group(1))
        if latest is None or value > latest:
            latest = value
    return latest


def _first_existing(paths: Iterable[str]) -> str:
    paths = list(paths)
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def _resolve_data_file(checkpoint_dir: str, raw_path: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(
        _first_existing(
            [
                raw_path,
                os.path.join(checkpoint_dir, raw_path),
                os.path.join(os.getcwd(), raw_path),
            ]
        )
    )


def resolve_multistream_checkpoint(
    checkpoint_dir: str,
    lib_path: Optional[str] = None,
    rank: Optional[int] = None,
    iteration: Optional[int] = None,
) -> ResolvedMultiStreamCheckpoint:
    """
    Resolve a rank-local multistream checkpoint.

    Supported layouts:
    - pccheck pipeline layout: ``checkpoint_dir/stage_{rank}/multistream_info.json``
    - DistributedMultiStreamCheckpoint layout:
      ``checkpoint_dir/iter_xxxxxxx/rank_{rank}.json`` + rank data file metadata
    - single-process layout: ``checkpoint_dir/multistream_info.json``
    """
    resolved_rank = _resolve_rank(rank)
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    # Pipeline-parallel pccheck layout.
    stage_dir = os.path.join(checkpoint_dir, f"stage_{resolved_rank}")
    info_path = os.path.join(stage_dir, "multistream_info.json")
    if os.path.exists(info_path):
        info = _read_json(info_path)
        state_path = os.path.join(stage_dir, "training_state.json")
        training_state = _read_json(state_path) if os.path.exists(state_path) else {}
        return ResolvedMultiStreamCheckpoint(
            chkpt_dir=stage_dir,
            metadata_file=info["metadata_file"],
            checkpoint_file=info["checkpoint_file"],
            lib_path=info.get("lib_path") or lib_path or "",
            rank=resolved_rank,
            training_state=training_state,
            # training_state.json 的 slot 只是提示；由 loader 读文件头已发布记录选版本
            parall_iter=None,
        )

    # Single-process multistream layout.
    info_path = os.path.join(checkpoint_dir, "multistream_info.json")
    if os.path.exists(info_path):
        info = _read_json(info_path)
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        training_state = _read_json(state_path) if os.path.exists(state_path) else {}
        return ResolvedMultiStreamCheckpoint(
            chkpt_dir=checkpoint_dir,
            metadata_file=info["metadata_file"],
            checkpoint_file=info["checkpoint_file"],
            lib_path=info.get("lib_path") or lib_path or "",
            rank=resolved_rank,
            training_state=training_state,
            # training_state.json 的 slot 只是提示；由 loader 读文件头已发布记录选版本
            parall_iter=None,
        )

    # DistributedMultiStreamCheckpoint layout.
    if iteration is None:
        iteration = _latest_iteration(checkpoint_dir)
    if iteration is not None:
        iter_dir = os.path.join(checkpoint_dir, f"iter_{int(iteration):07d}")
        rank_meta_path = os.path.join(iter_dir, f"rank_{resolved_rank}.json")
        if os.path.exists(rank_meta_path):
            rank_meta = _read_json(rank_meta_path)
            checkpoint_file = _resolve_data_file(checkpoint_dir, rank_meta["data_file"])
            metadata_file = checkpoint_file + ".metadata.json"
            training_state = dict(rank_meta)
            training_state.setdefault("completed_steps", int(iteration))
            training_state.setdefault("iteration", int(iteration))
            return ResolvedMultiStreamCheckpoint(
                chkpt_dir=os.path.dirname(checkpoint_file),
                metadata_file=metadata_file,
                checkpoint_file=checkpoint_file,
                lib_path=lib_path or "",
                rank=resolved_rank,
                training_state=training_state,
                parall_iter=rank_meta.get("slot"),
            )

    raise FileNotFoundError(
        "Could not resolve a multistream checkpoint for "
        f"rank={resolved_rank} under {checkpoint_dir}"
    )


class ForwardLoadGate:
    """
    Forward pre-hook gate that waits only for the model weights needed by each module.

    Hooks are removed as soon as every layer group is forward-ready. Optimizer
    state restore continues in the background and is guarded separately before
    the first optimizer update.
    """

    def __init__(self, model: nn.Module, loader: MultiStreamStateLoader) -> None:
        self.model = model
        self.loader = loader
        self._ready_chunks: Set[int] = set()
        self._handles: List[Any] = []
        self._removed = False
        self._lock = threading.Lock()
        self._root_fallback_param_count = 0
        self._root_fallback_param_examples: List[str] = []
        self._hook_module_count = 0
        self._install_hooks()

    def _install_hooks(self) -> None:
        named_modules = dict(self.model.named_modules())
        module_to_chunks: Dict[str, Set[int]] = {}

        # v3 语义：param_name_to_chunks 的值是 chunk 集合——split 分组下一个张量
        # 可能横跨多个 chunk，pre-hook 必须等它的全部覆盖 chunk（单值映射是旧 bug）。
        for param_name, chunk_ids in self.loader.param_name_to_chunks.items():
            module_name = param_name.rsplit(".", 1)[0] if "." in param_name else ""
            while module_name not in named_modules and "." in module_name:
                module_name = module_name.rsplit(".", 1)[0]
            if module_name not in named_modules:
                self._root_fallback_param_count += 1
                if len(self._root_fallback_param_examples) < 8:
                    self._root_fallback_param_examples.append(param_name)
                module_name = ""
            module_to_chunks.setdefault(module_name, set()).update(
                int(c) for c in chunk_ids)

        for module_name, chunk_ids in module_to_chunks.items():
            if not chunk_ids:
                continue
            module = named_modules[module_name]
            self._handles.append(module.register_forward_pre_hook(self._make_hook(tuple(sorted(chunk_ids)))))
        self._hook_module_count = len(self._handles)

    def _make_hook(self, chunk_ids: Tuple[int, ...]):
        def hook(_module: nn.Module, _inputs: Tuple[Any, ...]) -> None:
            self.wait_for_chunks(chunk_ids)

        return hook

    def wait_for_chunks(self, chunk_ids: Iterable[int]) -> None:
        if self._removed:
            return
        for chunk_idx in chunk_ids:
            if chunk_idx in self._ready_chunks:
                continue
            if self.loader.is_forward_group_ready(chunk_idx):
                self._ready_chunks.add(chunk_idx)
                continue
            self.loader.wait_for_forward_group(chunk_idx)
            if self.loader.is_forward_group_ready(chunk_idx):
                self._ready_chunks.add(chunk_idx)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "forward_hook_module_count": int(self._hook_module_count),
            "forward_hook_root_fallback_param_count": int(self._root_fallback_param_count),
            "forward_hook_root_fallback_param_examples": list(self._root_fallback_param_examples),
        }

    def remove(self) -> None:
        with self._lock:
            if self._removed:
                return
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._removed = True


def _reconcile_training_state(json_state, loader, rank):
    """以 extra blob 中的发布载荷为准（与数据同 slot 原子发布）；JSON 只作补充。
    多 rank 时核对各 rank 恢复到的 completed_steps 一致，否则拒绝恢复。"""
    state = dict(json_state or {})
    payload = loader.published_payload
    if payload:
        js = state.get("completed_steps")
        bs = payload.get("completed_steps")
        if js is not None and bs is not None and int(js) != int(bs):
            print(f"[WARN] rank {rank}: training_state.json completed_steps={js} != "
                  f"published blob {bs}; using the published value")
        state.update(payload)
    state["slot"] = int(loader.parall_iter)
    state["slot_source"] = loader.slot_source
    state["published_counter"] = loader.published_counter
    if torch.distributed.is_available() and torch.distributed.is_initialized() \
            and torch.distributed.get_world_size() > 1:
        mine = state.get("completed_steps")
        allv = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(allv, mine)
        if len({v for v in allv}) != 1:
            raise RuntimeError(
                f"inconsistent per-rank checkpoints: completed_steps by rank = {allv}; "
                f"stages published different steps (crash between per-stage publishes)")
    return state


class PipelayerAsyncLoadHandle:
    def __init__(
        self,
        loader: MultiStreamStateLoader,
        gate: Optional[ForwardLoadGate] = None,
        training_state: Optional[Dict[str, Any]] = None,
        rank: int = 0,
        poll_interval: float = 0.05,
    ) -> None:
        self.loader = loader
        self.gate = gate
        self.training_state = training_state or {}
        self.rank = int(rank)
        self.poll_interval = float(poll_interval)
        self._forward_complete_event = threading.Event()
        self._complete_event = threading.Event()
        self._closed = False
        self._close_lock = threading.Lock()
        self._optimizer_guard: Optional[Tuple[Any, str, Any]] = None
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        atexit.register(self.cleanup)

    def _monitor_loop(self) -> None:
        while not self._complete_event.is_set() and not self._closed:
            if not self._forward_complete_event.is_set() and self._all_forward_chunks_loaded():
                self._mark_forward_complete()
            if self._all_chunks_loaded():
                self._mark_complete()
                return
            time.sleep(self.poll_interval)

    def _all_forward_chunks_loaded(self) -> bool:
        return self.loader.is_forward_ready()

    def _all_chunks_loaded(self) -> bool:
        return self.loader.is_update_ready()

    def _mark_forward_complete(self) -> None:
        if self.gate is not None:
            self.gate.remove()
        self._forward_complete_event.set()

    def _mark_complete(self) -> None:
        self._mark_forward_complete()
        self._complete_event.set()

    def is_complete(self) -> bool:
        if self._complete_event.is_set():
            return True
        if self._all_chunks_loaded():
            self._mark_complete()
            return True
        return False

    def wait_for_chunk(self, chunk_idx: int) -> None:
        self.loader.wait_for_chunk(chunk_idx)

    def wait_for_forward_chunk(self, chunk_idx: int) -> None:
        self.loader.wait_for_forward_group(chunk_idx)

    def wait_forward_ready(self) -> None:
        if not self._forward_complete_event.is_set():
            self.loader.wait_forward_ready()
            self._mark_forward_complete()

    def wait_update_ready(self) -> None:
        if not self._complete_event.is_set():
            self.loader.wait_update_ready()
            self._mark_complete()

    def wait_all(self) -> None:
        self.wait_update_ready()

    def get_restore_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if hasattr(self.loader, "get_restore_stats"):
            stats.update(self.loader.get_restore_stats())
        if self.gate is not None and hasattr(self.gate, "get_stats"):
            stats.update(self.gate.get_stats())
        return stats

    def install_optimizer_step_guard(self, engine: Any, method_name: Optional[str] = None) -> None:
        target = engine
        if target is None:
            target = self.loader.optimizer
            method_name = method_name or "step"
        elif method_name is None:
            # PipelineEngine dispatches OptimizerStep through _INSTRUCTION_MAP and
            # binds the class method directly, so patch the lower-level model-step
            # method that is still reached by every optimizer update.
            method_name = "_take_model_step" if hasattr(target, "_take_model_step") else "_exec_optimizer_step"
        if not hasattr(target, method_name):
            return

        original = getattr(target, method_name)
        handle = self

        def guarded_model_step(*args: Any, **kwargs: Any) -> Any:
            handle.wait_update_ready()
            setattr(target, method_name, original)
            handle._optimizer_guard = None
            return original(*args, **kwargs)

        setattr(target, method_name, guarded_model_step)
        self._optimizer_guard = (target, method_name, original)

    def cleanup(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self._optimizer_guard is not None:
                engine, method_name, original = self._optimizer_guard
                setattr(engine, method_name, original)
                self._optimizer_guard = None
            if self.gate is not None:
                self.gate.remove()
            self.loader.stop()
            self._forward_complete_event.set()
            self._complete_event.set()


def start_distributed_multistream_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    lib_path: Optional[str] = None,
    rank: Optional[int] = None,
    iteration: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    engine: Any = None,
    install_forward_hooks: bool = True,
    guard_optimizer_step: bool = True,
    load_grad: bool = False,
    host_queue_maxsize: int = 32,
    num_copy_workers: Optional[int] = None,
    parall_iter: Optional[int] = None,
    forward_prefetch_window: int = 2,
    restore_schedule_policy: str = "window_interleave",
) -> PipelayerAsyncLoadHandle:
    resolved = resolve_multistream_checkpoint(
        checkpoint_dir=checkpoint_dir,
        lib_path=lib_path,
        rank=rank,
        iteration=iteration,
    )
    if not resolved.lib_path:
        raise ValueError("lib_path is required when it is not present in multistream_info.json")

    resolved_device = _resolve_device(device)
    loader = MultiStreamStateLoader(
        model=model,
        optimizer=optimizer,
        chkpt_dir=resolved.chkpt_dir,
        lib_path=resolved.lib_path,
        metadata_file=resolved.metadata_file,
        checkpoint_file=resolved.checkpoint_file,
        device=str(resolved_device),
        host_queue_maxsize=host_queue_maxsize,
        num_copy_workers=num_copy_workers,
        load_grad=load_grad,
        parall_iter=parall_iter if parall_iter is not None else resolved.parall_iter,
        forward_prefetch_window=forward_prefetch_window,
        restore_schedule_policy=restore_schedule_policy,
    )
    training_state = _reconcile_training_state(resolved.training_state, loader, resolved.rank)
    gate = ForwardLoadGate(model, loader) if install_forward_hooks else None
    handle = PipelayerAsyncLoadHandle(
        loader=loader,
        gate=gate,
        training_state=training_state,
        rank=resolved.rank,
    )
    if guard_optimizer_step:
        handle.install_optimizer_step_guard(engine)
    return handle
