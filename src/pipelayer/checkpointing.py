from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from ctypes import cdll, c_void_p, c_char_p, c_int, c_size_t, POINTER

import torch
import torch.nn as nn


@dataclass(frozen=True)
class _LayerGroupManifest:
    group_idx: int
    chunk_key: str
    forward_order: int
    layer_ids: Tuple[int, ...]
    param_names: Tuple[str, ...]
    chunk_meta: Dict[str, Any]


@dataclass(order=True)
class _RestoreTask:
    priority: int
    sequence: int
    group_idx: int = field(compare=False)
    phase: str = field(compare=False)
    streams: Tuple[str, ...] = field(compare=False)
    chunk_meta: Dict[str, Any] = field(compare=False)


@dataclass(order=True)
class _BufferedRestoreTask:
    priority: int
    sequence: int
    group_idx: int = field(compare=False)
    phase: str = field(compare=False)
    streams: Tuple[str, ...] = field(compare=False)
    buffers: Dict[str, torch.Tensor] = field(compare=False)
    chunk_meta: Dict[str, Any] = field(compare=False)


def save_model_chunked(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_dir: str,
    target_chunk_bytes: int = 50 * 1024 * 1024,
) -> int:
    """
    保存模型与优化器状态为多个 chunk。
    逻辑：
      1) 按参数名的第一层前缀分组（兼容模块语义，如 encoder/decoder/...）。
      2) 每组内按字节大小切分为多个子 chunk（默认目标 50MB）。
    同时保存：
      - 每个 chunk 的模型参数与对应的优化器 state（按参数名）。
      - optimizer 的 param_groups（将 param 的 id 替换为 name）。
      - metadata.json：包含 chunk 映射信息与数量。

    返回：chunk 数量。
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()

    # name <-> param <-> id 三向映射（用于提取优化器状态）
    param_name_to_param = {name: param for name, param in model.named_parameters()}
    id_to_name = {id(param): name for name, param in model.named_parameters()}
    param_to_optim_state: Dict[nn.Parameter, Dict[str, Any]] = {}

    for group in optim_state_dict.get("param_groups", []):
        for param_id in group.get("params", []):
            if param_id in optim_state_dict.get("state", {}) and param_id in id_to_name:
                name = id_to_name[param_id]
                param = param_name_to_param[name]
                param_to_optim_state[param] = optim_state_dict["state"][param_id]

    # 先按 prefix 分组（兼容你原来的做法）
    groups: Dict[str, List[str]] = defaultdict(list)
    for key in model_state_dict.keys():
        prefix = key.split(".", 1)[0] if "." in key else "__root__"
        groups[prefix].append(key)

    chunk_metadata: Dict[int, Dict[str, Any]] = {}
    chunk_idx = 0

    # helper: size of a tensor in bytes
    def tensor_size_bytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    # 对每个 prefix 内部按字节切分为若干子chunk
    for group_name, state_keys_in_group in groups.items():
        cur_chunk_keys: List[str] = []
        cur_chunk_bytes = 0
        for key in state_keys_in_group:
            t = model_state_dict[key]
            if not isinstance(t, torch.Tensor):
                # 对于非张量（极少数 buffer 情况），跳过
                continue
            b = tensor_size_bytes(t)

            # 如果加入后超过阈值且已有内容，则先 flush
            if cur_chunk_bytes + b > target_chunk_bytes and cur_chunk_keys:
                chunk_idx = _flush_chunk(
                    save_dir,
                    chunk_idx,
                    cur_chunk_keys,
                    model_state_dict,
                    param_name_to_param,
                    param_to_optim_state,
                    chunk_metadata,
                    group_name,
                )
                cur_chunk_keys = []
                cur_chunk_bytes = 0

            cur_chunk_keys.append(key)
            cur_chunk_bytes += b

            # 若单个参数本身就超过阈值，则立即写入（独立 chunk）
            if cur_chunk_bytes >= target_chunk_bytes:
                chunk_idx = _flush_chunk(
                    save_dir,
                    chunk_idx,
                    cur_chunk_keys,
                    model_state_dict,
                    param_name_to_param,
                    param_to_optim_state,
                    chunk_metadata,
                    group_name,
                )
                cur_chunk_keys = []
                cur_chunk_bytes = 0

        # flush 剩余
        if cur_chunk_keys:
            chunk_idx = _flush_chunk(
                save_dir,
                chunk_idx,
                cur_chunk_keys,
                model_state_dict,
                param_name_to_param,
                param_to_optim_state,
                chunk_metadata,
                group_name,
            )

    # 保存 optimizer param_groups（id -> name）
    new_param_groups = []
    for group in optim_state_dict.get("param_groups", []):
        new_group = dict(group)
        new_group["params"] = []
        for param_id in group.get("params", []):
            if param_id in id_to_name:
                new_group["params"].append(id_to_name[param_id])
        new_param_groups.append(new_group)
    torch.save(new_param_groups, os.path.join(save_dir, "optimizer_param_groups.pt"))

    # 保存 metadata（包含每个 chunk 的参数列表）
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"num_chunks": chunk_idx, "chunks": chunk_metadata}, f, indent=4, ensure_ascii=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Saver] Model ({total_params/1e6:.2f}M params) and Optimizer chunked into {chunk_idx} parts -> {save_dir}")
    if total_params < 100e6:
        print("[Warning] 模型参数太小，pipeline IO 优化效果可能不明显。")
    return chunk_idx


def _flush_chunk(
    save_dir: str,
    chunk_idx: int,
    cur_chunk_keys: List[str],
    model_state_dict: Dict[str, torch.Tensor],
    param_name_to_param: Dict[str, nn.Parameter],
    param_to_optim_state: Dict[nn.Parameter, Dict[str, Any]],
    chunk_metadata: Dict[int, Dict[str, Any]],
    group_name: str,
) -> int:
    # 构造模型子状态
    chunk_model_sd = {k: model_state_dict[k].cpu() for k in cur_chunk_keys if isinstance(model_state_dict[k], torch.Tensor)}
    # 构造优化器子状态（按参数名）
    chunk_optim_sd_states: Dict[str, Dict[str, Any]] = {}
    for k in cur_chunk_keys:
        if k in param_name_to_param:
            param_tensor = param_name_to_param[k]
            if param_tensor in param_to_optim_state:
                chunk_optim_sd_states[k] = param_to_optim_state[param_tensor]
    chunk_file = f"chunk_{chunk_idx}.pt"
    torch.save({"model": chunk_model_sd, "optimizer_states": chunk_optim_sd_states}, os.path.join(save_dir, chunk_file))
    chunk_metadata[chunk_idx] = {"file": chunk_file, "params": list(cur_chunk_keys), "group": group_name}
    return chunk_idx + 1


class PipelinedStateLoader:
    """
    修复后的流水加载器，避免 CUDA 隐式同步问题，并支持 CPU 回退路径。
    - 独立 IO 线程：按 chunk 顺序读盘 -> pin memory -> 入队。
    - 多个拷贝工作线程：从 pinned 内存到 device，更新模型参数与优化器 state。
    - 每个 chunk 一个事件，用于前向或外部等待。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chkpt_dir: str,
        device: str = "cuda:0",
        host_queue_maxsize: int = 32,
        num_copy_workers: Optional[int] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.chkpt_dir = chkpt_dir
        self.device = torch.device(device)
        # 注意：model 通常已由 PipelayerModelWrapper 移动到目标设备
        # 此处仅检查而非强制移动，避免冗余操作
        if next(model.parameters(), None) is not None:
            model_device = next(model.parameters()).device
            if model_device != self.device:
                self.model.to(self.device)

        with open(os.path.join(chkpt_dir, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.num_chunks = int(self.metadata["num_chunks"])

        # host queue（producer -> 多 copy worker）
        self.host_buffer_queue: "queue.Queue[Any]" = queue.Queue(maxsize=host_queue_maxsize)

        # 参数与 buffer 名 -> 张量映射（用于原位写入）
        self.param_name_to_tensor: Dict[str, nn.Parameter] = {name: p for name, p in self.model.named_parameters()}
        self.buffer_name_to_tensor: Dict[str, torch.Tensor] = {name: b for name, b in self.model.named_buffers()}

        # 构建 param_name -> chunk_idx 的反向索引（便于前向按模块等待所需 chunk）
        self.param_name_to_chunk: Dict[str, int] = {}
        for cid, meta in self.metadata["chunks"].items():
            ic = int(cid)
            for pname in meta["params"]:
                self.param_name_to_chunk[pname] = ic

        # 加载优化器 param groups
        self._load_optimizer_param_groups()

        # 设置 copy workers 数量
        if num_copy_workers is None:
            # 保守：默认 1，避免多流复杂同步问题；可按需调大
            num_copy_workers = 1
        self.num_copy_workers = int(num_copy_workers)

        # 创建 events/streams（CUDA 或 CPU 回退）
        self._init_device_primitives()

        self.producer_stop_event = threading.Event()
        self.copy_worker_stop_event = threading.Event()

        # 启动线程
        self._producer_thread = threading.Thread(target=self._io_producer_loop, daemon=True)
        self._copy_worker_threads: List[threading.Thread] = []
        self._producer_thread.start()
        for i in range(self.num_copy_workers):
            t = threading.Thread(target=self._copy_worker_loop, args=(i,), daemon=True)
            t.start()
            self._copy_worker_threads.append(t)

        # 清理
        atexit.register(self.stop)

    def _init_device_primitives(self) -> None:
        self._chunk_loaded_flags = [False for _ in range(self.num_chunks)]
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.copy_streams = [torch.cuda.Stream() for _ in range(self.num_copy_workers)]
                self.events = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_chunks)]
        else:
            # CPU 回退：无 stream 概念，使用 None 占位；事件用 threading.Event
            self.copy_streams = [None for _ in range(self.num_copy_workers)]
            self.events = [threading.Event() for _ in range(self.num_chunks)]

    def _load_optimizer_param_groups(self) -> None:
        param_groups_path = os.path.join(self.chkpt_dir, "optimizer_param_groups.pt")
        if os.path.exists(param_groups_path):
            saved_param_groups = torch.load(param_groups_path, map_location="cpu")
            new_param_groups = []
            for saved_group in saved_param_groups:
                new_group = dict(saved_group)
                new_params = []
                for pname in saved_group.get("params", []):
                    if pname in self.param_name_to_tensor:
                        new_params.append(self.param_name_to_tensor[pname])
                new_group["params"] = new_params
                if new_group["params"]:
                    new_param_groups.append(new_group)
            self.optimizer.param_groups = new_param_groups
            self.optimizer.state = defaultdict(dict)

    def _io_producer_loop(self) -> None:
        """IO 线程：按 chunk 顺序从磁盘 load 到 CPU 并 pin memory，然后放 queue"""
        try:
            for i in range(self.num_chunks):
                if self.producer_stop_event.is_set():
                    break
                fname = self.metadata["chunks"][str(i)]["file"]
                chunk_path = os.path.join(self.chkpt_dir, fname)
                chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)

                # pin model tensors
                pinned_model = {}
                for k, v in chunk_data["model"].items():
                    if isinstance(v, torch.Tensor):
                        pinned_model[k] = v.pin_memory()
                    else:
                        pinned_model[k] = v

                optim_states = chunk_data.get("optimizer_states", {})

                # pin optimizer tensor states too
                for param_name, state in optim_states.items():
                    for state_name, tensor in list(state.items()):
                        if isinstance(tensor, torch.Tensor):
                            state[state_name] = tensor.pin_memory()

                # push to queue
                self.host_buffer_queue.put((i, pinned_model, optim_states))
        except Exception as e:
            print(f"[ERROR] Producer thread error: {e}")
            self.producer_stop_event.set()

    def _copy_worker_loop(self, worker_idx: int) -> None:
        """每个 copy worker 使用独立流，把 CPU pinned tensor 上传到设备并写入模型/optimizer"""
        stream = self.copy_streams[worker_idx]
        try:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    self._copy_worker_loop_impl(worker_idx, stream)  # type: ignore[arg-type]
            else:
                self._copy_worker_loop_impl(worker_idx, stream=None)
        except Exception as e:
            print(f"[ERROR] Copy worker thread error (idx={worker_idx}): {e}")
            self.copy_worker_stop_event.set()

    def _copy_worker_loop_impl(self, worker_idx: int) -> None:
        while not (self.copy_worker_stop_event.is_set() and self.host_buffer_queue.empty()):
            try:
                item = self.host_buffer_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self.host_buffer_queue.task_done()
                break
            chunk_idx, host_model_chunk, host_optim_chunk = item
            try:
                if self.device.type == "cuda":
                    stream = (
                        self.update_copy_streams[worker_idx]
                        if task.phase == "update"
                        else self.weight_copy_streams[worker_idx]
                    )
                    assert stream is not None
                    with torch.cuda.stream(stream):
                        self._apply_model_chunk(host_model_chunk)
                        self._apply_optimizer_chunk(host_optim_chunk)
                        self.events[chunk_idx].record(stream)  # type: ignore[union-attr]
                else:
                    # CPU：直接赋值并置位事件
                    self._apply_model_chunk(host_model_chunk)
                    self._apply_optimizer_chunk(host_optim_chunk)
                    self.events[chunk_idx].set()  # type: ignore[union-attr]
            finally:
                self.host_buffer_queue.task_done()

    def _apply_model_chunk(self, host_model_chunk: Dict[str, torch.Tensor]) -> None:
        for name, host_tensor in host_model_chunk.items():
            if not isinstance(host_tensor, torch.Tensor):
                continue
            # 参数优先
            if name in self.param_name_to_tensor:
                param = self.param_name_to_tensor[name]
                param.data.copy_(host_tensor.to(self.device, non_blocking=self.device.type == "cuda"))
            elif name in self.buffer_name_to_tensor:
                buf = self.buffer_name_to_tensor[name]
                buf.data.copy_(host_tensor.to(self.device, non_blocking=self.device.type == "cuda"))
            # 未知条目忽略（例如某些临时 buffer）

    def _apply_optimizer_chunk(self, host_optim_chunk: Dict[str, Dict[str, Any]]) -> None:
        for param_name, state_dict in host_optim_chunk.items():
            if param_name not in self.param_name_to_tensor:
                continue
            param_tensor = self.param_name_to_tensor[param_name]
            new_state: Dict[str, Any] = {}
            for state_name, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    new_state[state_name] = tensor.to(self.device, non_blocking=self.device.type == "cuda")
                else:
                    new_state[state_name] = tensor
            self.optimizer.state[param_tensor] = new_state

    def stop(self) -> None:
        """停止并释放资源：优雅关闭 producer 与 copy workers"""
        self.producer_stop_event.set()
        self.copy_worker_stop_event.set()
        # 唤醒可能阻塞的 workers
        for _ in range(self.num_copy_workers + 2):
            try:
                self.host_buffer_queue.put_nowait(None)
            except queue.Full:
                break

        if hasattr(self, "_producer_thread") and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)
        for t in getattr(self, "_copy_worker_threads", []):
            if t.is_alive():
                t.join(timeout=2.0)

        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # 清空队列
        while not self.host_buffer_queue.empty():
            try:
                self.host_buffer_queue.get_nowait()
                self.host_buffer_queue.task_done()
            except queue.Empty:
                break

    def wait_for_chunk(self, chunk_idx: int) -> None:
        """阻塞等待某个 chunk 的上传（供 forward 使用）"""
        if 0 <= chunk_idx < self.num_chunks:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.current_stream().wait_event(self.events[chunk_idx])  # type: ignore[arg-type]
            else:
                self.events[chunk_idx].wait()  # type: ignore[union-attr]

    def is_chunk_loaded(self, chunk_idx: int) -> bool:
        """非阻塞查询某个 chunk 是否已加载完成（用于 monitor 或 progress）"""
        if 0 <= chunk_idx < self.num_chunks:
            if self.device.type == "cuda":
                return bool(self.events[chunk_idx].query())  # type: ignore[union-attr]
            else:
                return bool(self.events[chunk_idx].is_set())  # type: ignore[union-attr]
        return False

    def debug_stats(self) -> None:
        print(f"[Loader Stats] Queue Size: {self.host_buffer_queue.qsize()}, CopyWorkers: {self.num_copy_workers}")


class MultiStreamStateLoader:
    """
    基于 multistream 格式的流水加载器（C++ mmap 读取 + Python pipeline）。

    - C++ 侧负责 mmap + 按 stream/offset 拷贝到 CPU pinned buffer。
    - Python 侧负责异步拷贝到 device，并原位更新参数与优化器状态。
    - 恢复状态拆成 forward-ready（param）和 update-ready（grad/Adam state）。
    - 调度采用 window interleave：保持 forward weights 预取窗口，同时穿插 optimizer state。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        chkpt_dir: str,
        lib_path: str,
        checkpoint_file: Optional[str] = None,
        metadata_file: str = "multistream_metadata.json",
        device: str = "cuda:0",
        host_queue_maxsize: int = 32,
        num_copy_workers: Optional[int] = None,
        load_grad: bool = False,
        parall_iter: Optional[int] = None,
        forward_prefetch_window: int = 2,
        restore_schedule_policy: str = "window_interleave",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.chkpt_dir = chkpt_dir
        self.device = torch.device(device)
        # 注意：model 通常已由 PipelayerModelWrapper 移动到目标设备
        # 此处仅检查而非强制移动，避免冗余操作
        if next(model.parameters(), None) is not None:
            model_device = next(model.parameters()).device
            if model_device != self.device:
                self.model.to(self.device)
        self.load_grad = load_grad
        self.forward_prefetch_window = max(0, int(forward_prefetch_window))
        self.restore_schedule_policy = str(restore_schedule_policy)
        supported_policies = {"window_interleave", "weights_first", "critical_stream_first"}
        if self.restore_schedule_policy not in supported_policies:
            raise ValueError(f"Unsupported restore_schedule_policy={restore_schedule_policy!r}")
        self._fatal_error: Optional[BaseException] = None

        metadata_path = metadata_file
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(chkpt_dir, metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        metadata_num_chunks = int(self.metadata["num_chunks"])
        self.stream_names: List[str] = list(self.metadata.get("stream_names", ["param", "grad", "exp_avg", "exp_avg_sq"]))
        self.stream_sizes: List[int] = [int(x) for x in self.metadata.get("stream_sizes", [])]
        self.max_async = int(self.metadata.get("max_async", 1))
        self.stream_name_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(self.stream_names)}
        self._validate_stream_layout()

        if checkpoint_file is None:
            checkpoint_file = self.metadata.get("checkpoint_file")
        if checkpoint_file is None:
            raise ValueError("checkpoint_file is required (not found in metadata).")
        if not os.path.isabs(checkpoint_file):
            checkpoint_file = os.path.join(chkpt_dir, checkpoint_file)
        self.checkpoint_file = checkpoint_file

        # 参数与 buffer 名 -> 张量映射（用于原位写入）
        self.param_name_to_tensor: Dict[str, nn.Parameter] = {name: p for name, p in self.model.named_parameters()}
        self.buffer_name_to_tensor: Dict[str, torch.Tensor] = {name: b for name, b in self.model.named_buffers()}

        self.layer_group_manifest = self._build_layer_group_manifest(metadata_num_chunks)
        self.group_idx_to_manifest: Dict[int, _LayerGroupManifest] = {
            group.group_idx: group for group in self.layer_group_manifest
        }
        self.num_chunks = len(self.layer_group_manifest)

        # param_name -> forward/update group idx. Keep the old name for callers.
        self.param_name_to_chunk: Dict[str, int] = {}
        self.param_name_to_group = self.param_name_to_chunk
        for group in self.layer_group_manifest:
            for pname in group.param_names:
                if pname in self.param_name_to_chunk:
                    raise ValueError(f"Duplicate parameter '{pname}' in multistream layer-group metadata.")
                self.param_name_to_chunk[pname] = int(group.group_idx)

        # 异步加载前先校验：避免在 copy worker 里晚失败且错误不直观。
        self._validate_metadata_compatibility()

        # 加载 optimizer param groups（若提供）
        self._load_optimizer_param_groups()
        self.optimizer_steps: Dict[str, int] = self.metadata.get("optimizer_steps", {})
        
        # 验证 optimizer step 一致性，并计算全局 step
        self.global_optimizer_step: Optional[int] = self._validate_and_get_global_step()

        if num_copy_workers is None:
            num_copy_workers = 1
        self.num_copy_workers = int(num_copy_workers)
        self._required_update_streams = self._build_required_update_streams()
        self._restore_start_time = time.time()
        self._stats_lock = threading.Lock()
        self._restore_stats: Dict[str, Any] = {
            "forward_wait_s": 0.0,
            "update_barrier_wait_s": 0.0,
            "forward_ready_all_s": None,
            "update_ready_all_s": None,
            "forward_group_ready_s": {},
            "update_group_ready_s": {},
            "forward_group_wait_s": {},
            "update_group_wait_s": {},
            "restore_schedule_policy": self.restore_schedule_policy,
            "forward_prefetch_window": self.forward_prefetch_window,
            "required_update_streams": list(self._required_update_streams),
        }

        self._init_device_primitives()
        self._init_reader(lib_path)

        # 优先从 metadata 读取 latest_parall_iter，避免调用可能崩溃的 C++ 接口
        if parall_iter is None:
            parall_iter = self.metadata.get("latest_parall_iter")
        if parall_iter is None:
            # C++ 的 get_latest_parall_iter 存在段错误风险，直接默认使用 0
            # 对于旧版本 metadata（没有 latest_parall_iter 字段），使用 0 是安全的
            print("[WARN] latest_parall_iter not found in metadata, defaulting to 0")
            parall_iter = 0
        self.parall_iter = int(parall_iter)
        print(f"[MultiStreamStateLoader] Using parall_iter={self.parall_iter}")

        self.host_buffer_queue: "queue.PriorityQueue[Any]" = queue.PriorityQueue(maxsize=host_queue_maxsize)
        self.producer_stop_event = threading.Event()
        self.copy_worker_stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._optimizer_state_lock = threading.Lock()
        self._restore_tasks = self._build_restore_tasks()
        self._record_restore_task_stats()

        self._producer_thread = threading.Thread(target=self._io_producer_loop, daemon=True)
        self._copy_worker_threads: List[threading.Thread] = []
        self._producer_thread.start()
        for i in range(self.num_copy_workers):
            t = threading.Thread(target=self._copy_worker_loop, args=(i,), daemon=True)
            t.start()
            self._copy_worker_threads.append(t)

        atexit.register(self.stop)

    def _set_fatal_error(self, err: BaseException) -> None:
        self._fatal_error = err

    def _raise_if_fatal_error(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                f"MultiStreamStateLoader encountered a fatal error: {self._fatal_error}"
            ) from self._fatal_error

    def _validate_stream_layout(self) -> None:
        if len(set(self.stream_names)) != len(self.stream_names):
            raise ValueError(f"Multistream metadata contains duplicate stream names: {self.stream_names}")

        if self.stream_sizes and len(self.stream_sizes) != len(self.stream_names):
            raise ValueError(
                "Multistream metadata stream_sizes length must match stream_names length: "
                f"stream_sizes={len(self.stream_sizes)}, stream_names={len(self.stream_names)}"
            )

        supported_streams = {"param", "grad", "exp_avg", "exp_avg_sq"}
        unknown_streams = sorted(set(self.stream_names) - supported_streams)
        if unknown_streams:
            raise ValueError(
                "Unsupported multistream checkpoint streams: "
                f"{unknown_streams}. Supported streams are {sorted(supported_streams)}."
            )

        required_streams = {"param", "exp_avg", "exp_avg_sq"}
        if self.load_grad:
            required_streams.add("grad")
        missing_streams = sorted(required_streams - set(self.stream_names))
        if missing_streams:
            raise ValueError(
                "Multistream checkpoint is missing required streams: "
                f"{missing_streams}. stream_names={self.stream_names}"
            )

    def _build_layer_group_manifest(self, metadata_num_chunks: int) -> List[_LayerGroupManifest]:
        chunks = self.metadata.get("chunks")
        if not isinstance(chunks, dict) or not chunks:
            raise ValueError("Multistream metadata must contain a non-empty 'chunks' object.")

        groups: List[_LayerGroupManifest] = []
        for raw_chunk_key, chunk_meta in chunks.items():
            if not isinstance(chunk_meta, dict):
                raise ValueError(f"Chunk metadata for key {raw_chunk_key!r} must be an object.")

            try:
                group_idx = int(chunk_meta.get("group_idx", raw_chunk_key))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid group_idx for chunk {raw_chunk_key!r}: {chunk_meta.get('group_idx')!r}") from exc

            params = chunk_meta.get("params", [])
            if not isinstance(params, list) or not params:
                raise ValueError(f"Layer group {group_idx} must contain a non-empty params list.")

            stream_slices = chunk_meta.get("stream_slices", {})
            if not isinstance(stream_slices, dict):
                raise ValueError(f"Layer group {group_idx} must contain stream_slices metadata.")
            for stream_name in ("param", "exp_avg", "exp_avg_sq"):
                slice_info = stream_slices.get(stream_name)
                if not slice_info or int(slice_info.get("size", 0)) <= 0:
                    raise ValueError(f"Layer group {group_idx} is missing non-empty {stream_name!r} stream slice.")
            if self.load_grad:
                grad_slice = stream_slices.get("grad")
                if not grad_slice or int(grad_slice.get("size", 0)) <= 0:
                    raise ValueError("load_grad=True but layer group " f"{group_idx} has no non-empty 'grad' slice.")

            layer_ids = tuple(int(x) for x in chunk_meta.get("layer_ids", []))
            param_names: List[str] = []
            for pinfo in params:
                name = pinfo.get("name") if isinstance(pinfo, dict) else None
                if not name:
                    raise ValueError(f"Layer group {group_idx} contains a parameter entry without a name.")
                param_names.append(str(name))

            forward_anchor = min(layer_ids) if layer_ids else group_idx
            groups.append(
                _LayerGroupManifest(
                    group_idx=group_idx,
                    chunk_key=str(raw_chunk_key),
                    forward_order=int(forward_anchor),
                    layer_ids=layer_ids,
                    param_names=tuple(param_names),
                    chunk_meta=chunk_meta,
                )
            )

        if len(groups) != metadata_num_chunks:
            raise ValueError(
                "Multistream metadata num_chunks does not match chunks entries: "
                f"num_chunks={metadata_num_chunks}, chunks={len(groups)}"
            )

        actual_group_ids = {group.group_idx for group in groups}
        expected_group_ids = set(range(len(groups)))
        if actual_group_ids != expected_group_ids:
            raise ValueError(
                "Multistream layer-group ids must be contiguous from 0 to num_chunks-1. "
                f"actual={sorted(actual_group_ids)}, expected={sorted(expected_group_ids)}"
            )

        return sorted(groups, key=lambda group: (group.forward_order, group.group_idx))

    def _build_restore_tasks(self) -> List[_RestoreTask]:
        tasks: List[_RestoreTask] = []
        sequence = 0
        num_groups = len(self.layer_group_manifest)

        def add_weight_task(group: _LayerGroupManifest) -> None:
            nonlocal sequence
            tasks.append(
                _RestoreTask(
                    priority=sequence,
                    sequence=sequence,
                    group_idx=group.group_idx,
                    phase="weights",
                    streams=("param",),
                    chunk_meta=group.chunk_meta,
                )
            )
            sequence += 1

        def add_update_task(group: _LayerGroupManifest, stream_name: str) -> None:
            nonlocal sequence
            tasks.append(
                _RestoreTask(
                    priority=sequence,
                    sequence=sequence,
                    group_idx=group.group_idx,
                    phase="update",
                    streams=(stream_name,),
                    chunk_meta=group.chunk_meta,
                )
            )
            sequence += 1

        if self.restore_schedule_policy in {"weights_first", "critical_stream_first"}:
            for group in self.layer_group_manifest:
                add_weight_task(group)
            for group in self.layer_group_manifest:
                for stream_name in self._required_update_streams:
                    add_update_task(group, stream_name)
            return tasks

        next_weight_idx = 0
        initial_weight_count = min(num_groups, self.forward_prefetch_window + 1)
        while next_weight_idx < initial_weight_count:
            add_weight_task(self.layer_group_manifest[next_weight_idx])
            next_weight_idx += 1

        for update_group in self.layer_group_manifest:
            for stream_name in self._required_update_streams:
                add_update_task(update_group, stream_name)
                if next_weight_idx < num_groups:
                    add_weight_task(self.layer_group_manifest[next_weight_idx])
                    next_weight_idx += 1

        while next_weight_idx < num_groups:
            add_weight_task(self.layer_group_manifest[next_weight_idx])
            next_weight_idx += 1

        return tasks

    def _record_restore_task_stats(self) -> None:
        order = [
            {
                "sequence": int(task.sequence),
                "phase": task.phase,
                "group_idx": int(task.group_idx),
                "streams": list(task.streams),
            }
            for task in self._restore_tasks
        ]
        with self._stats_lock:
            self._restore_stats["num_restore_tasks"] = len(order)
            self._restore_stats["restore_task_order"] = order

    def _build_required_update_streams(self) -> Tuple[str, ...]:
        streams = ["exp_avg", "exp_avg_sq"]
        if self.load_grad:
            streams.append("grad")
        return tuple(streams)

    def _validate_metadata_compatibility(self) -> None:
        """校验 metadata 中参数与当前模型参数的可兼容性（按 name + numel）。"""
        common_count = 0
        mismatches: List[Tuple[str, int, int, Tuple[int, ...]]] = []

        for chunk in self.metadata.get("chunks", {}).values():
            for pinfo in chunk.get("params", []):
                name = pinfo.get("name")
                if not name or name not in self.param_name_to_tensor:
                    continue
                common_count += 1
                expected_numel = int(pinfo.get("numel", -1))
                param = self.param_name_to_tensor[name]
                actual_numel = int(param.numel())
                if expected_numel > 0 and expected_numel != actual_numel:
                    mismatches.append((name, expected_numel, actual_numel, tuple(param.shape)))

        if common_count == 0:
            raise ValueError(
                "No overlapping parameter names between multistream metadata and current model. "
                "Checkpoint/model pair is incompatible."
            )

        if mismatches:
            preview = "; ".join(
                [
                    f"{n}: ckpt_numel={e}, model_numel={a}, model_shape={s}"
                    for n, e, a, s in mismatches[:5]
                ]
            )
            raise ValueError(
                "Multistream checkpoint tensor shapes do not match current model. "
                "Please use the same architecture/config used at save time. "
                f"Examples: {preview}"
            )

    def _init_reader(self, lib_path: str) -> None:
        self.lib = cdll.LoadLibrary(lib_path)

        self.lib.reader.restype = c_void_p
        self.lib.reader.argtypes = [c_char_p, c_int]

        self.lib.init_streams.restype = c_int
        self.lib.init_streams.argtypes = [c_void_p, c_int, POINTER(c_size_t)]

        self.lib.read_stream_chunk.restype = None
        self.lib.read_stream_chunk.argtypes = [c_void_p, c_int, c_void_p, c_size_t, c_size_t, c_int]

        self.lib.get_latest_parall_iter.restype = c_int
        self.lib.get_latest_parall_iter.argtypes = [c_void_p]

        self.lib.close_writer.restype = None
        self.lib.close_writer.argtypes = [c_void_p]

        self.reader_obj = self.lib.reader(self.checkpoint_file.encode("utf-8"), int(self.max_async))

        if self.stream_sizes:
            stream_sizes_arr = (c_size_t * len(self.stream_sizes))(*self.stream_sizes)
            ret = self.lib.init_streams(self.reader_obj, int(len(self.stream_sizes)), stream_sizes_arr)
            if ret != 0:
                raise RuntimeError("Failed to initialize streams in multistream reader")

    def _init_device_primitives(self) -> None:
        self._forward_ready_flags = [False for _ in range(self.num_chunks)]
        self._update_ready_flags = [False for _ in range(self.num_chunks)]
        self._update_stream_ready_flags = [
            {stream_name: False for stream_name in self._required_update_streams}
            for _ in range(self.num_chunks)
        ]
        self._chunk_loaded_flags = self._update_ready_flags
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                low_priority = 0
                high_priority = 0
                try:
                    low_priority, high_priority = torch.cuda.get_stream_priority_range()
                except Exception:
                    pass
                self.weight_copy_streams = [
                    torch.cuda.Stream(priority=high_priority) for _ in range(self.num_copy_workers)
                ]
                self.update_copy_streams = [
                    torch.cuda.Stream(priority=low_priority) for _ in range(self.num_copy_workers)
                ]
                self.copy_streams = self.weight_copy_streams
                self.forward_events = [torch.cuda.Event(enable_timing=False) for _ in range(self.num_chunks)]
                self.update_stream_events = [
                    {stream_name: torch.cuda.Event(enable_timing=False) for stream_name in self._required_update_streams}
                    for _ in range(self.num_chunks)
                ]
                self.update_events = self.update_stream_events
                self.events = self.update_events
        else:
            self.weight_copy_streams = [None for _ in range(self.num_copy_workers)]
            self.update_copy_streams = [None for _ in range(self.num_copy_workers)]
            self.copy_streams = self.weight_copy_streams
            self.forward_events = [threading.Event() for _ in range(self.num_chunks)]
            self.update_stream_events = [
                {stream_name: threading.Event() for stream_name in self._required_update_streams}
                for _ in range(self.num_chunks)
            ]
            self.update_events = self.update_stream_events
            self.events = self.update_events

    def _load_optimizer_param_groups(self) -> None:
        groups = self.metadata.get("optimizer_param_groups")
        if not groups:
            return
        new_param_groups = []
        for group in groups:
            new_group = dict(group)
            new_params = []
            for pname in group.get("params", []):
                if pname in self.param_name_to_tensor:
                    new_params.append(self.param_name_to_tensor[pname])
            new_group["params"] = new_params
            if new_group["params"]:
                new_param_groups.append(new_group)
        if new_param_groups:
            self.optimizer.param_groups = new_param_groups
            self.optimizer.state = defaultdict(dict)

    def _validate_and_get_global_step(self) -> Optional[int]:
        """
        获取全局 optimizer step 值。
        
        性能优先：直接取第一个 step 值，假设保存侧已保证一致性。
        正常训练流程中所有参数的 step 始终一致。
        """
        if not self.optimizer_steps:
            fallback = int(self.metadata.get("latest_parall_iter", 0))
            return fallback if fallback > 0 else None
        # O(1) 取任意一个值即可
        step = int(next(iter(self.optimizer_steps.values())))
        if step > 0:
            return step
        # 某些旧/异常 metadata 会把 optimizer_steps 写成全 0，但优化器状态并非初始化态。
        # 这会在 AdamW foreach 中触发 bias_correction1=0 导致除零。
        fallback = int(self.metadata.get("latest_parall_iter", 0))
        if fallback > 0:
            print(
                f"[WARN] optimizer_steps in metadata is non-positive ({step}); "
                f"fallback to latest_parall_iter={fallback}."
            )
            return fallback
        print(
            f"[WARN] optimizer_steps in metadata is non-positive ({step}) and latest_parall_iter is unavailable; "
            "fallback optimizer step to 1."
        )
        return 1

    def _add_restore_stat(self, key: str, value: float) -> None:
        with self._stats_lock:
            current = self._restore_stats.get(key, 0.0)
            self._restore_stats[key] = float(current) + float(value)

    def _add_group_restore_stat(self, key: str, group_idx: int, value: float) -> None:
        with self._stats_lock:
            group_stats = self._restore_stats.setdefault(key, {})
            current = group_stats.get(group_idx, 0.0)
            group_stats[group_idx] = float(current) + float(value)

    def _mark_forward_group_ready(self, group_idx: int) -> None:
        elapsed = time.time() - self._restore_start_time
        with self._state_lock:
            self._forward_ready_flags[group_idx] = True
            with self._stats_lock:
                self._restore_stats["forward_group_ready_s"][group_idx] = elapsed
                if self._restore_stats["forward_ready_all_s"] is None and all(self._forward_ready_flags):
                    self._restore_stats["forward_ready_all_s"] = elapsed

    def _mark_update_stream_ready(self, group_idx: int, stream_name: str) -> None:
        elapsed = time.time() - self._restore_start_time
        with self._state_lock:
            self._update_stream_ready_flags[group_idx][stream_name] = True
            group_ready = all(self._update_stream_ready_flags[group_idx].values())
            if group_ready:
                self._update_ready_flags[group_idx] = True
            with self._stats_lock:
                if group_ready:
                    self._restore_stats["update_group_ready_s"][group_idx] = elapsed
                    if self._restore_stats["update_ready_all_s"] is None and all(self._update_ready_flags):
                        self._restore_stats["update_ready_all_s"] = elapsed

    def get_restore_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._restore_stats)
            stats["forward_group_ready_s"] = dict(self._restore_stats["forward_group_ready_s"])
            stats["update_group_ready_s"] = dict(self._restore_stats["update_group_ready_s"])
            stats["forward_group_wait_s"] = dict(self._restore_stats.get("forward_group_wait_s", {}))
            stats["update_group_wait_s"] = dict(self._restore_stats.get("update_group_wait_s", {}))
            stats["required_update_streams"] = list(self._restore_stats.get("required_update_streams", []))
            stats["restore_task_order"] = list(self._restore_stats.get("restore_task_order", []))
            return stats

    def _io_producer_loop(self) -> None:
        try:
            for task in self._restore_tasks:
                if self.producer_stop_event.is_set():
                    break
                buffers: Dict[str, torch.Tensor] = {}
                for stream_name in task.streams:
                    if self.producer_stop_event.is_set():
                        break
                    slice_info = task.chunk_meta.get("stream_slices", {}).get(stream_name)
                    if not slice_info:
                        raise ValueError(
                            f"Layer group {task.group_idx} is missing required stream slice {stream_name!r}."
                        )
                    size = int(slice_info.get("size", 0))
                    if size == 0:
                        continue
                    offset = int(slice_info["offset"])
                    buffer = torch.empty(size, dtype=torch.float32, pin_memory=self.device.type == "cuda")
                    reader_obj = self.reader_obj
                    if reader_obj is None:
                        raise RuntimeError("reader_obj is None while producer is still running")
                    stream_idx = self.stream_name_to_idx[stream_name]
                    self.lib.read_stream_chunk(
                        reader_obj,
                        int(stream_idx),
                        c_void_p(buffer.data_ptr()),
                        c_size_t(offset),
                        c_size_t(size),
                        c_int(self.parall_iter),
                    )
                    buffers[stream_name] = buffer

                if self.producer_stop_event.is_set():
                    break
                self.host_buffer_queue.put(
                    _BufferedRestoreTask(
                        priority=task.priority,
                        sequence=task.sequence,
                        group_idx=task.group_idx,
                        phase=task.phase,
                        streams=task.streams,
                        buffers=buffers,
                        chunk_meta=task.chunk_meta,
                    )
                )
        except Exception as e:
            print(f"[ERROR] MultiStream producer error: {e}")
            self._set_fatal_error(e)
            self.producer_stop_event.set()
            self.copy_worker_stop_event.set()
        finally:
            for i in range(self.num_copy_workers):
                self.host_buffer_queue.put(
                    _BufferedRestoreTask(
                        priority=10_000_000,
                        sequence=10_000_000 + i,
                        group_idx=-1,
                        phase="stop",
                        streams=(),
                        buffers={},
                        chunk_meta={},
                    )
                )

    def _copy_worker_loop(self, worker_idx: int) -> None:
        try:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    self._copy_worker_loop_impl(worker_idx)
            else:
                self._copy_worker_loop_impl(worker_idx)
        except Exception as e:
            print(f"[ERROR] MultiStream copy worker error (idx={worker_idx}): {e}")
            self._set_fatal_error(e)
            self.producer_stop_event.set()
            self.copy_worker_stop_event.set()

    def _copy_worker_loop_impl(self, worker_idx: int) -> None:
        while True:
            if self.copy_worker_stop_event.is_set() and self.host_buffer_queue.empty():
                break
            try:
                item = self.host_buffer_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item.phase == "stop":
                self.host_buffer_queue.task_done()
                break
            task = item
            try:
                if self.device.type == "cuda":
                    stream = (
                        self.update_copy_streams[worker_idx]
                        if task.phase == "update"
                        else self.weight_copy_streams[worker_idx]
                    )
                    assert stream is not None
                    with torch.cuda.stream(stream):
                        self._apply_buffered_restore_task(task)
                        if task.phase == "weights":
                            self.forward_events[task.group_idx].record(stream)  # type: ignore[union-attr]
                        elif task.phase == "update":
                            for stream_name in task.streams:
                                self.update_stream_events[task.group_idx][stream_name].record(stream)  # type: ignore[union-attr]
                        else:
                            raise ValueError(f"Unknown restore task phase: {task.phase!r}")
                    if task.phase == "weights":
                        self._mark_forward_group_ready(task.group_idx)
                    elif task.phase == "update":
                        for stream_name in task.streams:
                            self._mark_update_stream_ready(task.group_idx, stream_name)
                else:
                    self._apply_buffered_restore_task(task)
                    if task.phase == "weights":
                        self.forward_events[task.group_idx].set()  # type: ignore[union-attr]
                        self._mark_forward_group_ready(task.group_idx)
                    elif task.phase == "update":
                        for stream_name in task.streams:
                            self.update_stream_events[task.group_idx][stream_name].set()  # type: ignore[union-attr]
                            self._mark_update_stream_ready(task.group_idx, stream_name)
                    else:
                        raise ValueError(f"Unknown restore task phase: {task.phase!r}")
            finally:
                self.host_buffer_queue.task_done()

    def _apply_buffered_restore_task(self, task: _BufferedRestoreTask) -> None:
        if task.phase == "weights":
            self._apply_weight_stream(task.buffers, task.chunk_meta)
        elif task.phase == "update":
            self._apply_update_streams(task.buffers, task.chunk_meta)
        else:
            raise ValueError(f"Unknown restore task phase: {task.phase!r}")

    def _apply_weight_stream(self, buffers: Dict[str, torch.Tensor], chunk_meta: Dict[str, Any]) -> None:
        for pinfo in chunk_meta.get("params", []):
            name = pinfo.get("name")
            if not name or name not in self.param_name_to_tensor:
                continue
            param = self.param_name_to_tensor[name]
            numel = int(pinfo.get("numel", param.numel()))
            if numel != int(param.numel()):
                raise ValueError(
                    f"Parameter numel mismatch for '{name}': ckpt={numel}, model={param.numel()}, "
                    f"model_shape={tuple(param.shape)}"
                )
            offsets = pinfo.get("offsets_in_chunk", {})
            dtype = param.dtype
            non_blocking = self.device.type == "cuda"

            if "param" not in buffers or "param" not in offsets:
                raise ValueError(f"Layer group is missing param data for '{name}'.")
            host_view = self._host_view_for_param(buffers, offsets, "param", name, numel, param.shape)
            param.data.copy_(host_view.to(self.device, dtype=dtype, non_blocking=non_blocking))

    def _apply_update_streams(self, buffers: Dict[str, torch.Tensor], chunk_meta: Dict[str, Any]) -> None:
        restore_streams = tuple(buffers.keys())
        for pinfo in chunk_meta.get("params", []):
            name = pinfo.get("name")
            if not name or name not in self.param_name_to_tensor:
                continue
            param = self.param_name_to_tensor[name]
            numel = int(pinfo.get("numel", param.numel()))
            if numel != int(param.numel()):
                raise ValueError(
                    f"Parameter numel mismatch for '{name}': ckpt={numel}, model={param.numel()}, "
                    f"model_shape={tuple(param.shape)}"
                )
            offsets = pinfo.get("offsets_in_chunk", {})
            dtype = param.dtype
            non_blocking = self.device.type == "cuda"

            if "grad" in restore_streams:
                if "grad" not in buffers or "grad" not in offsets:
                    raise ValueError(f"load_grad=True but checkpoint grad data is missing for '{name}'.")
                host_view = self._host_view_for_param(buffers, offsets, "grad", name, numel, param.shape)
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                param.grad.data.copy_(host_view.to(self.device, dtype=dtype, non_blocking=non_blocking))

            with self._optimizer_state_lock:
                state = self.optimizer.state.setdefault(param, {})
            for state_name in ("exp_avg", "exp_avg_sq"):
                if state_name in restore_streams:
                    self._restore_optimizer_tensor_state(
                        state, buffers, offsets, state_name, name, numel, param.shape, dtype
                    )
            self._restore_optimizer_step(state, name)

    def _host_view_for_param(
        self,
        buffers: Dict[str, torch.Tensor],
        offsets: Dict[str, Any],
        stream_name: str,
        param_name: str,
        numel: int,
        shape: torch.Size,
    ) -> torch.Tensor:
        start = int(offsets[stream_name])
        host_slice = buffers[stream_name][start:start + numel]
        if int(host_slice.numel()) != numel:
            raise ValueError(
                f"{stream_name} slice out-of-range for '{param_name}': requested [{start}:{start + numel}], "
                f"available={buffers[stream_name].numel()}"
            )
        return host_slice.view(shape)

    def _restore_optimizer_tensor_state(
        self,
        state: Dict[str, Any],
        buffers: Dict[str, torch.Tensor],
        offsets: Dict[str, Any],
        state_name: str,
        param_name: str,
        numel: int,
        shape: torch.Size,
        dtype: torch.dtype,
    ) -> None:
        if state_name not in buffers or state_name not in offsets:
            raise ValueError(f"Checkpoint optimizer state {state_name!r} is missing for '{param_name}'.")
        host_view = self._host_view_for_param(buffers, offsets, state_name, param_name, numel, shape)
        desired = host_view.to(self.device, dtype=dtype, non_blocking=self.device.type == "cuda")
        with self._optimizer_state_lock:
            existing = state.get(state_name)
            if (
                isinstance(existing, torch.Tensor)
                and existing.shape == desired.shape
                and existing.device == desired.device
                and existing.dtype == desired.dtype
            ):
                existing.data.copy_(desired)
                state[state_name] = existing
            else:
                state[state_name] = desired

    def _restore_optimizer_step(self, state: Dict[str, Any], param_name: str) -> None:
        step_value = None
        if self.global_optimizer_step is not None:
            step_value = float(self.global_optimizer_step)
        elif param_name in self.optimizer_steps:
            step_value = float(self.optimizer_steps[param_name])
        if step_value is None:
            return
        if step_value <= 0:
            fallback = float(int(self.metadata.get("latest_parall_iter", 0)))
            step_value = fallback if fallback > 0 else 1.0
        with self._optimizer_state_lock:
            existing_step = state.get("step")
            if isinstance(existing_step, torch.Tensor):
                step_dtype = existing_step.dtype if existing_step.dtype in (torch.float32, torch.float64) else torch.float32
                step_device = existing_step.device if existing_step.device.type == "cpu" else torch.device("cpu")
                existing_step.data.copy_(torch.tensor(step_value, dtype=step_dtype, device=step_device))
                state["step"] = existing_step
            else:
                state["step"] = torch.tensor(step_value, dtype=torch.float32, device="cpu")

    def stop(self) -> None:
        self.producer_stop_event.set()
        self.copy_worker_stop_event.set()
        for i in range(self.num_copy_workers + 2):
            try:
                self.host_buffer_queue.put_nowait(
                    _BufferedRestoreTask(
                        priority=10_000_000,
                        sequence=20_000_000 + i,
                        group_idx=-1,
                        phase="stop",
                        streams=(),
                        buffers={},
                        chunk_meta={},
                    )
                )
            except queue.Full:
                break

        producer_alive = False
        if hasattr(self, "_producer_thread") and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)
            producer_alive = self._producer_thread.is_alive()
        for t in getattr(self, "_copy_worker_threads", []):
            if t.is_alive():
                t.join(timeout=2.0)

        if getattr(self, "reader_obj", None) is not None:
            if producer_alive:
                # producer 还未退出时不能把 reader_obj 置空，否则可能在 C++ 层空指针崩溃。
                print("[WARN] producer thread still alive during stop(); skip reader handle cleanup this round.")
            else:
                try:
                    # reader_obj 由 self.lib.reader() 创建，是 reader 类型对象。
                    # close_writer 用于关闭 writer 对象，不能用于 reader，否则会段错误。
                    # 检查库是否导出了 close_reader；若没有则跳过，依赖进程退出时 mmap 自动释放。
                    if hasattr(self.lib, "close_reader"):
                        self.lib.close_reader.restype = None
                        self.lib.close_reader.argtypes = [c_void_p]
                        self.lib.close_reader(self.reader_obj)
                except Exception:
                    pass
                self.reader_obj = None

        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        while not self.host_buffer_queue.empty():
            try:
                self.host_buffer_queue.get_nowait()
                self.host_buffer_queue.task_done()
            except queue.Empty:
                break

    def _validate_group_idx(self, group_idx: int) -> None:
        if not 0 <= group_idx < self.num_chunks:
            raise IndexError(f"Layer group index out of range: {group_idx}")

    def _wait_until_event_recorded(self, flags: List[bool], group_idx: int) -> None:
        self._validate_group_idx(group_idx)
        while not flags[group_idx]:
            self._raise_if_fatal_error()
            time.sleep(0.001)
        self._raise_if_fatal_error()

    def _wait_for_group_event(self, events: List[Any], flags: List[bool], group_idx: int) -> None:
        self._raise_if_fatal_error()
        self._validate_group_idx(group_idx)
        if self.device.type == "cuda":
            self._wait_until_event_recorded(flags, group_idx)
            with torch.cuda.device(self.device):
                torch.cuda.current_stream().wait_event(events[group_idx])  # type: ignore[arg-type]
        else:
            while not events[group_idx].wait(timeout=0.05):  # type: ignore[union-attr]
                self._raise_if_fatal_error()
        self._raise_if_fatal_error()

    def _is_group_event_ready(self, events: List[Any], flags: List[bool], group_idx: int) -> bool:
        self._raise_if_fatal_error()
        self._validate_group_idx(group_idx)
        if self.device.type == "cuda":
            return bool(flags[group_idx]) and bool(events[group_idx].query())  # type: ignore[union-attr]
        return bool(flags[group_idx]) and bool(events[group_idx].is_set())  # type: ignore[union-attr]

    def wait_for_forward_group(self, group_idx: int) -> None:
        t0 = time.time()
        self._wait_for_group_event(self.forward_events, self._forward_ready_flags, group_idx)
        elapsed = time.time() - t0
        self._add_restore_stat("forward_wait_s", elapsed)
        self._add_group_restore_stat("forward_group_wait_s", group_idx, elapsed)

    def is_forward_group_ready(self, group_idx: int) -> bool:
        return self._is_group_event_ready(self.forward_events, self._forward_ready_flags, group_idx)

    def wait_for_update_group(self, group_idx: int) -> None:
        t0 = time.time()
        self._raise_if_fatal_error()
        self._validate_group_idx(group_idx)
        if self.device.type == "cuda":
            self._wait_until_event_recorded(self._update_ready_flags, group_idx)
            with torch.cuda.device(self.device):
                for stream_name in self._required_update_streams:
                    torch.cuda.current_stream().wait_event(  # type: ignore[arg-type]
                        self.update_stream_events[group_idx][stream_name]
                    )
        else:
            for stream_name in self._required_update_streams:
                event = self.update_stream_events[group_idx][stream_name]
                while not event.wait(timeout=0.05):  # type: ignore[union-attr]
                    self._raise_if_fatal_error()
        self._raise_if_fatal_error()
        self._add_group_restore_stat("update_group_wait_s", group_idx, time.time() - t0)

    def is_update_group_ready(self, group_idx: int) -> bool:
        self._raise_if_fatal_error()
        self._validate_group_idx(group_idx)
        if not self._update_ready_flags[group_idx]:
            return False
        if self.device.type == "cuda":
            return all(
                bool(self.update_stream_events[group_idx][stream_name].query())  # type: ignore[union-attr]
                for stream_name in self._required_update_streams
            )
        return all(
            bool(self.update_stream_events[group_idx][stream_name].is_set())  # type: ignore[union-attr]
            for stream_name in self._required_update_streams
        )

    def wait_forward_ready(self) -> None:
        for group_idx in range(self.num_chunks):
            self.wait_for_forward_group(group_idx)

    def is_forward_ready(self) -> bool:
        return all(self.is_forward_group_ready(group_idx) for group_idx in range(self.num_chunks))

    def wait_update_ready(self) -> None:
        t0 = time.time()
        for group_idx in range(self.num_chunks):
            self.wait_for_update_group(group_idx)
        self._add_restore_stat("update_barrier_wait_s", time.time() - t0)

    def is_update_ready(self) -> bool:
        return all(self.is_update_group_ready(group_idx) for group_idx in range(self.num_chunks))

    def wait_for_chunk(self, chunk_idx: int) -> None:
        """Backward-compatible full-restore wait: waits for update-ready."""
        self.wait_for_update_group(chunk_idx)

    def is_chunk_loaded(self, chunk_idx: int) -> bool:
        """Backward-compatible full-restore query: true only after update-ready."""
        return self.is_update_group_ready(chunk_idx)

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
