from __future__ import annotations

import atexit
import importlib.util
import json
import os
import queue
import random
import shutil
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, Any, Callable, List, Optional, Tuple

from ctypes import cdll, c_void_p, c_char_p, c_int, c_size_t, POINTER

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# v3 格式权威：保存侧 multistream_checkpoint.py 是布局/量化/blob 格式的唯一定义，
# 恢复端 import 复用（piece_extent/_quant_unit_floats/parse_extra_state_blob），
# 避免格式出现第三份拷贝。查找顺序：常规 import → PCCHECK_MS_DIR → 默认路径。
# ---------------------------------------------------------------------------
_MS_FORMAT_DIR_ENV = "PCCHECK_MS_DIR"
_MS_FORMAT_DEFAULT_DIR = "/home/linzhicheng/code/pccheck/checkpoint_eval/pccheck"
_ms_format_module = None


def _ms_format():
    """加载保存侧格式模块 multistream_checkpoint（惰性、进程内单例）。"""
    global _ms_format_module
    if _ms_format_module is None:
        # 优先复用调用方已导入的保存端模块（包形式导入时 sys.path 上没有顶层模块，
        # 旧逻辑会回退到默认目录，可能加载另一份、版本不同的格式代码）
        pkg_mod = sys.modules.get("checkpoint_eval.pccheck.multistream_checkpoint")
        try:
            if pkg_mod is not None:
                m = pkg_mod
            else:
                import multistream_checkpoint as m  # 调用方已注入 sys.path 的情形
        except ImportError:
            ms_dir = os.environ.get(_MS_FORMAT_DIR_ENV, _MS_FORMAT_DEFAULT_DIR)
            path = os.path.join(ms_dir, "multistream_checkpoint.py")
            if not os.path.exists(path):
                raise ImportError(
                    f"multistream_checkpoint.py not found at {path}; "
                    f"set {_MS_FORMAT_DIR_ENV} to the pccheck directory")
            spec = importlib.util.spec_from_file_location("multistream_checkpoint", path)
            m = importlib.util.module_from_spec(spec)
            sys.modules["multistream_checkpoint"] = m
            spec.loader.exec_module(m)
        _ms_format_module = m
        print(f"[pipelayer] checkpoint format module: {getattr(m, '__file__', m)}")
    return _ms_format_module


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
            self.optimizer.state = {}

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

    def _copy_worker_loop_impl(self, worker_idx: int, stream: Optional["torch.cuda.Stream"]) -> None:  # type: ignore[name-defined]
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
        load_grad: bool = False,          # v3 无 grad 流，保留参数仅为兼容旧调用方
        parall_iter: Optional[int] = None,
        master_applier: Optional[Callable] = None,  # (param, flat_slice, ps, pe) -> None
        weights_first: Optional[bool] = None,       # None → 读 PIPELAYER_WEIGHTS_FIRST
        forward_prefetch_window: int = 2,           # distributed 分支调度参数；v3 的
        restore_schedule_policy: str = "window_interleave",  # weights-first 两阶段调度取代其语义，仅保兼容
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
        if load_grad:
            print("[WARN] load_grad is deprecated: v3 checkpoints carry no grad stream "
                  "(grads are recomputed after resume); ignored.")
        self.master_applier = master_applier
        self._master_warned = False
        if weights_first is None:
            weights_first = os.environ.get("PIPELAYER_WEIGHTS_FIRST", "0") == "1"
        self.weights_first = bool(weights_first)
        self.forward_prefetch_window = int(forward_prefetch_window)
        self.restore_schedule_policy = str(restore_schedule_policy)
        self._fatal_error: Optional[BaseException] = None

        metadata_path = metadata_file
        if not os.path.isabs(metadata_path):
            metadata_path = os.path.join(chkpt_dir, metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # 版本门禁：v3 变更了数据区基址((max_async+4))与量化布局，旧格式本就不可读，
        # 不保留死代码兼容分支。
        version = int(self.metadata.get("version", 1))
        if version < 3:
            raise ValueError(
                f"multistream metadata version {version} is not supported by this loader "
                f"(requires >= 3: interleaved quant layout + extra stream + fixed data base). "
                f"Re-save the checkpoint with the current CascadeSave.")

        self.num_chunks = int(self.metadata["num_chunks"])
        self.stream_names: List[str] = list(self.metadata.get("stream_names", ["param", "master", "exp_avg", "exp_avg_sq"]))
        self.stream_sizes: List[int] = [int(x) for x in self.metadata.get("stream_sizes", [])]
        self.max_async = int(self.metadata.get("max_async", 1))

        qm = self.metadata.get("quant_moments", {}) or {}
        self.quant_enabled = bool(qm.get("enabled"))
        self.quant_block = int(qm.get("block", 0))
        es = self.metadata.get("extra_state", {}) or {}
        self.extra_state_enabled = bool(es.get("enabled"))
        self.extra_stream_id = int(es.get("stream_id", 4))
        self.extra_stream_floats = int(es.get("floats", 0))
        # 数据流 = 除 extra 外的流（extra 由 load_extra_state 单独读取）
        self.data_stream_names = [s for s in self.stream_names if s != "extra"]

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

        # param_name -> 覆盖它的全部 chunk（split 分组下张量可横跨多组，
        # forward pre-hook 必须等待全部覆盖 chunk——正确性要求非优化）。
        # param_name_to_chunk（单值）保留作兼容别名（取最后一个）。
        self.param_name_to_chunks: Dict[str, set] = {}
        self.param_name_to_chunk: Dict[str, int] = {}
        for cid, meta in self.metadata.get("chunks", {}).items():
            ic = int(cid)
            for pinfo in meta.get("params", []):
                pname = pinfo.get("name")
                if pname:
                    self.param_name_to_chunks.setdefault(pname, set()).add(ic)
                    self.param_name_to_chunk[pname] = ic

        # 异步加载前先校验：避免在 copy worker 里晚失败且错误不直观。
        self._validate_metadata_compatibility()

        # 加载 optimizer param groups（若提供）
        self._load_optimizer_param_groups()
        self.optimizer_steps: Dict[str, int] = self.metadata.get("optimizer_steps", {})

        # 验证 optimizer step 一致性，并计算全局 step
        # （blob 中的 optimizer_step 更可靠，slot 解析后若可读则覆盖此值）
        self.global_optimizer_step: Optional[int] = self._validate_and_get_global_step()

        # master 流非空但未提供 applier：只能跳过（4N 对照档的 grad 流按
        # param_groups 序排列，与 named_parameters 序不同，不支持按层恢复）。
        master_idx = self.data_stream_names.index("master") if "master" in self.data_stream_names else -1
        self._master_stream_floats = self.stream_sizes[master_idx] if master_idx >= 0 else 0
        if self._master_stream_floats > 0 and self.master_applier is None:
            print("[WARN] checkpoint has a non-empty master stream but no master_applier "
                  "was provided; master stream will be skipped during restore.")

        if num_copy_workers is None:
            # state 已预建（无分配/竞态障碍），2 个 worker 让 H2D 与反量化重叠
            num_copy_workers = 2
        self.num_copy_workers = int(num_copy_workers)

        self._init_device_primitives()
        self._init_reader(lib_path)
        self._init_pinned_pool()

        # 选版本：以文件头的已发布记录为准（C++ 在 durability barrier + CAS 之后持久化，
        # 且先更新它才回收旧 slot）。metadata 的 latest_parall_iter / training_state 的
        # slot 写在发布之外的时刻，崩溃时可能指向在途或已复用的 slot，只作旧格式回退。
        self.published_counter: Optional[int] = None
        self.slot_source = "explicit"
        if parall_iter is None:
            parall_iter, self.published_counter, self.slot_source = self._resolve_published_slot()
        self.parall_iter = int(parall_iter)
        print(f"[MultiStreamStateLoader] Using parall_iter={self.parall_iter} "
              f"(source={self.slot_source}, counter={self.published_counter})")

        # slot 已定 → 先读 extra blob：其中的 optimizer_step 是 finalize 时刻的
        # 真实值（metadata 的 optimizer_steps 存在"槽号回填"历史 kludge），
        # 用它修正预建 state 的 step；顺带完成 crc/parall_iter 一致性早失败。
        self._extra_state_early: Optional[Dict[str, Any]] = None
        if self.extra_state_enabled:
            try:
                self._extra_state_early = self.load_extra_state()
                blob_step = (self._extra_state_early or {}).get("optimizer_step")
                if blob_step is not None and float(blob_step) > 0:
                    self.global_optimizer_step = int(float(blob_step))
                    print(f"[MultiStreamStateLoader] optimizer step from extra blob: "
                          f"{self.global_optimizer_step}")
            except Exception as e:
                if self.slot_source == "published":
                    # 已发布 slot 的 blob 校验失败 = 数据损坏，不能带病恢复
                    raise RuntimeError(
                        f"extra-state blob of published slot {self.parall_iter} failed "
                        f"verification: {e}") from e
                print(f"[WARN] early extra-state read failed: {e}")

        # 预建全部 optimizer state（zeros + step）：copy worker 热路径零分配、
        # 多 worker 无 dict 竞态、apply 无 dtype/device 分支。
        self._precreate_optimizer_state()

        self.host_buffer_queue: "queue.Queue[Any]" = queue.Queue(maxsize=host_queue_maxsize)
        self.producer_stop_event = threading.Event()
        self.copy_worker_stop_event = threading.Event()

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
        n = self.num_chunks
        # 两级就绪：param（forward 可跑）与 full（optimizer 状态齐）。
        # weights-first 关闭时二者在同一 phase 一起置位。
        self._chunk_param_loaded = [False] * n
        self._chunk_full_loaded = [False] * n
        # CPU 侧就绪信号：wait_for_chunk* 先等它再做 GPU 侧 wait_event——
        # 未 record 的 CUDA event 上 wait_event 是空操作，原实现存在
        # "hook 先于 worker 记录事件" 的潜伏时序漏洞，这里补上。
        self._param_done_cpu = [threading.Event() for _ in range(n)]
        self._full_done_cpu = [threading.Event() for _ in range(n)]
        self._param_ready_time: Dict[int, float] = {}
        self._full_ready_time: Dict[int, float] = {}
        self._loader_init_time = time.time()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self.copy_streams = [torch.cuda.Stream() for _ in range(self.num_copy_workers)]
                self.param_events = [torch.cuda.Event(enable_timing=False) for _ in range(n)]
                self.full_events = [torch.cuda.Event(enable_timing=False) for _ in range(n)]
        else:
            self.copy_streams = [None for _ in range(self.num_copy_workers)]
            self.param_events = self._param_done_cpu
            self.full_events = self._full_done_cpu
        # 兼容旧属性名
        self.events = self.param_events
        self._chunk_loaded_flags = self._chunk_param_loaded

    def _precreate_optimizer_state(self) -> None:
        """为全部待恢复参数预建 exp_avg/exp_avg_sq（零张量）与 step。

        copy worker 热路径因此零分配、零 dict 竞态；apply 只做原位片段填充
        （split 分组下同一参数的不同 piece 可能由不同 worker 并发写入，
        预建后各写各的 disjoint 区间，天然无竞态）。
        """
        step_value = float(self.global_optimizer_step) if self.global_optimizer_step else 1.0
        if step_value <= 0:
            step_value = 1.0
        with torch.no_grad():
            for name, param in self.param_name_to_tensor.items():
                if name not in self.param_name_to_chunks:
                    continue
                state = self.optimizer.state.setdefault(param, {})
                if not isinstance(state.get("exp_avg"), torch.Tensor):
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                if not isinstance(state.get("exp_avg_sq"), torch.Tensor):
                    state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                # AdamW foreach 要求 step 为 CPU float tensor
                state["step"] = torch.tensor(step_value, dtype=torch.float32, device="cpu")

    def _init_pinned_pool(self) -> None:
        """按"各数据流最大 chunk span"预分配 pinned buffer 池。

        替代热路径逐 chunk torch.empty(pin_memory=True)（cudaHostAlloc 串行化，
        已诊断瓶颈）；池深有限 ⇒ 对 producer 形成天然背压，读盘不会无界超前。
        """
        per_stream = max(1, int(os.environ.get("PIPELAYER_PIN_BUFFERS_PER_STREAM", "2")))
        max_span: Dict[str, int] = {}
        for chunk in self.metadata.get("chunks", {}).values():
            for sname, sl in chunk.get("stream_slices", {}).items():
                max_span[sname] = max(max_span.get(sname, 0), int(sl.get("size", 0)))
        pin = self.device.type == "cuda"
        self._pinned_pool: Dict[str, "queue.Queue[torch.Tensor]"] = {}
        total_bytes = 0
        for sname, span in max_span.items():
            if span <= 0:
                continue
            if sname == "master" and self.master_applier is None:
                continue  # 不消费的流不占 pinned 内存
            pool: "queue.Queue[torch.Tensor]" = queue.Queue()
            for _ in range(per_stream):
                pool.put(torch.empty(span, dtype=torch.float32, pin_memory=pin))
                total_bytes += span * 4
            self._pinned_pool[sname] = pool
        sizes_desc = {k: f"{per_stream}x{max_span[k] * 4 / 1e6:.0f}MB" for k in self._pinned_pool}
        print(f"[MultiStreamStateLoader] pinned pool: {sizes_desc} total {total_bytes / 1e9:.2f} GB")

    def _pool_get(self, stream_name: str) -> torch.Tensor:
        pool = self._pinned_pool[stream_name]
        while True:
            try:
                return pool.get(timeout=0.2)
            except queue.Empty:
                if self.producer_stop_event.is_set():
                    raise RuntimeError("producer stopped while waiting for pinned buffer")

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
            self.optimizer.state = {}

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

    # phase → 本阶段要读的数据流。weights-first：先全组 param（forward 起跑），
    # 再回头补 master+moments（首个 optimizer.step 前由 wait_all_full 兜底）。
    _PHASE_STREAMS = {
        "param": ("param",),
        "opt": ("master", "exp_avg", "exp_avg_sq"),
        "full": ("param", "master", "exp_avg", "exp_avg_sq"),
    }

    def _read_chunk_streams(self, chunk_meta: Dict[str, Any], phase: str) -> Dict[str, Tuple[torch.Tensor, int]]:
        """把本 chunk 指定 phase 的各流 span 从 mmap 读入池化 pinned buffer。

        Returns: {stream_name: (pooled_buffer, used_floats)}
        """
        out: Dict[str, Tuple[torch.Tensor, int]] = {}
        for stream_name in self._PHASE_STREAMS[phase]:
            if stream_name not in self._pinned_pool:
                continue  # 空流（如 fp32 下 master）或无 applier 的 master
            slice_info = chunk_meta.get("stream_slices", {}).get(stream_name)
            if not slice_info or int(slice_info.get("size", 0)) == 0:
                continue
            size = int(slice_info["size"])
            offset = int(slice_info["offset"])
            buf = self._pool_get(stream_name)
            assert size <= buf.numel(), (
                f"stream {stream_name} slice {size} exceeds pool buffer {buf.numel()}")
            reader_obj = self.reader_obj
            if reader_obj is None:
                raise RuntimeError("reader_obj is None while producer is still running")
            self.lib.read_stream_chunk(
                reader_obj,
                int(self.stream_names.index(stream_name)),
                c_void_p(buf.data_ptr()),
                c_size_t(offset),
                c_size_t(size),
                c_int(self.parall_iter),
            )
            out[stream_name] = (buf, size)
        return out

    def _io_producer_loop(self) -> None:
        try:
            phases = ("param", "opt") if self.weights_first else ("full",)
            for phase in phases:
                for i in range(self.num_chunks):
                    if self.producer_stop_event.is_set():
                        return
                    chunk_meta = self.metadata["chunks"][str(i)]
                    buffers = self._read_chunk_streams(chunk_meta, phase)
                    self.host_buffer_queue.put((i, phase, buffers, chunk_meta))
        except Exception as e:
            if self.producer_stop_event.is_set():
                return  # 正常 stop 流程中被打断（如 _pool_get 退出），非错误
            print(f"[ERROR] MultiStream producer error: {e}")
            self._set_fatal_error(e)
            self.producer_stop_event.set()

    def _copy_worker_loop(self, worker_idx: int) -> None:
        stream = self.copy_streams[worker_idx]
        try:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    self._copy_worker_loop_impl(worker_idx, stream)  # type: ignore[arg-type]
            else:
                self._copy_worker_loop_impl(worker_idx, stream=None)
        except Exception as e:
            print(f"[ERROR] MultiStream copy worker error (idx={worker_idx}): {e}")
            self._set_fatal_error(e)
            self.copy_worker_stop_event.set()

    def _copy_worker_loop_impl(self, worker_idx: int, stream: Optional["torch.cuda.Stream"]) -> None:  # type: ignore[name-defined]
        while not (self.copy_worker_stop_event.is_set() and self.host_buffer_queue.empty()):
            try:
                item = self.host_buffer_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self.host_buffer_queue.task_done()
                break
            chunk_idx, phase, buffers, chunk_meta = item
            try:
                views = {k: b[:sz] for k, (b, sz) in buffers.items()}
                if self.device.type == "cuda":
                    assert stream is not None
                    with torch.cuda.stream(stream):
                        self._apply_multistream_chunk(views, chunk_meta, phase)
                        if phase in ("param", "full"):
                            self.param_events[chunk_idx].record(stream)  # type: ignore[union-attr]
                        if phase in ("opt", "full"):
                            self.full_events[chunk_idx].record(stream)  # type: ignore[union-attr]
                    # 归还 pinned buffer 前必须确认本 stream 上的 H2D 已完成；
                    # 顺带保证置位 CPU 就绪信号时 CUDA event 必已 record+complete
                    stream.synchronize()
                else:
                    self._apply_multistream_chunk(views, chunk_meta, phase)
                self._mark_phase_done(chunk_idx, phase)
            finally:
                for k, (b, _) in buffers.items():
                    self._pinned_pool[k].put(b)
                self.host_buffer_queue.task_done()

    def _mark_phase_done(self, chunk_idx: int, phase: str) -> None:
        now = time.time() - self._loader_init_time
        if phase in ("param", "full"):
            self._chunk_param_loaded[chunk_idx] = True
            self._param_ready_time.setdefault(chunk_idx, now)
            self._param_done_cpu[chunk_idx].set()
        if phase in ("opt", "full"):
            self._chunk_full_loaded[chunk_idx] = True
            self._full_ready_time.setdefault(chunk_idx, now)
            self._full_done_cpu[chunk_idx].set()

    def _apply_multistream_chunk(self, buffers: Dict[str, torch.Tensor],
                                 chunk_meta: Dict[str, Any], phase: str = "full") -> None:
        """piece 粒度 apply：本 chunk 的每个 pinfo 只覆盖 elem_range=[ps,pe)。

        param/moment 一律经 flat 视图片段原位写入（pinned→device 单跳直拷，
        不再经 .to(device) 临时张量）；量化 moment 走 GPU 反量化；master 流
        交给 master_applier 钩子。step/param_groups 已在 __init__ 预建，
        不在热路径处理。split 分组下同一参数的不同 piece 可能由不同 worker
        并发写入 disjoint 区间，预建 state 后天然无竞态。
        """
        non_blocking = self.device.type == "cuda"
        with torch.no_grad():
            for pinfo in chunk_meta.get("params", []):
                name = pinfo.get("name")
                if not name or name not in self.param_name_to_tensor:
                    continue
                param = self.param_name_to_tensor[name]
                numel = int(pinfo.get("numel", param.numel()))
                if numel != int(param.numel()):
                    raise ValueError(
                        f"Parameter numel mismatch for '{name}': ckpt={numel}, "
                        f"model={param.numel()}, model_shape={tuple(param.shape)}")
                ps, pe = [int(x) for x in pinfo.get("elem_range", (0, numel))]
                offsets = pinfo.get("offsets_in_chunk", {})
                sizes = pinfo.get("sizes", {})
                seg = pe - ps
                if seg <= 0:
                    continue

                if phase in ("param", "full") and "param" in buffers:
                    start = int(offsets["param"])
                    src = buffers["param"][start:start + seg]
                    if int(src.numel()) != seg:
                        raise ValueError(
                            f"param slice out-of-range for '{name}': "
                            f"[{start}:{start + seg}] in {buffers['param'].numel()}")
                    param.data.view(-1)[ps:pe].copy_(src, non_blocking=non_blocking)

                if phase not in ("opt", "full"):
                    continue

                if ("master" in buffers and int(sizes.get("master", 0)) > 0
                        and self.master_applier is not None):
                    start = int(offsets["master"])
                    src = buffers["master"][start:start + seg]
                    self.master_applier(param, src, ps, pe)

                state = self.optimizer.state[param]
                q = pinfo.get("quant")
                for skey, sqrt_domain in (("exp_avg", False), ("exp_avg_sq", True)):
                    if skey not in buffers:
                        continue
                    start = int(offsets[skey])
                    dst = state[skey].view(-1)[ps:pe]
                    if q:
                        fsz = int(sizes[skey])
                        src = buffers[skey][start:start + fsz]
                        self._dequant_piece_into(dst, src, q, ps, pe, sqrt_domain)
                    else:
                        src = buffers[skey][start:start + seg]
                        if int(src.numel()) != seg:
                            raise ValueError(
                                f"{skey} slice out-of-range for '{name}': "
                                f"[{start}:{start + seg}] in {buffers[skey].numel()}")
                        dst.copy_(src, non_blocking=non_blocking)

    def _dequant_piece_into(self, dst_flat: torch.Tensor, src_region: torch.Tensor,
                            quant_meta: Dict[str, Any], ps: int, pe: int,
                            sqrt_domain: bool) -> None:
        """v3 交错单元反量化：src_region 为若干单元 [int8*B | fp32 scale] 顺排。

        与保存侧 _pack_moment_int8 严格互逆：block 是张量内局部坐标、切口
        block 对齐 ⇒ 分段反量化与整张量反量化逐位一致。packed 字节直接 H2D
        后在 GPU 上解交错+反量化（moment 流传输量为 fp32 的 ~1/4）。
        """
        fmt = quant_meta.get("format")
        if fmt != "interleaved-block":
            raise ValueError(f"unsupported quant format {fmt!r} "
                             f"(this loader supports 'interleaved-block' only)")
        block = int(quant_meta["block"])
        unit = block // 4 + 1
        n = int(quant_meta["numel"])
        assert ps % block == 0 and (pe == n or pe % block == 0), (
            f"quant piece [{ps},{pe}) not aligned to block {block} (numel={n})")
        nb_piece = (pe + block - 1) // block - ps // block
        expected = nb_piece * unit
        if int(src_region.numel()) != expected:
            raise ValueError(f"quant region size {src_region.numel()} != expected {expected}")
        region = src_region.to(self.device, non_blocking=self.device.type == "cuda")
        region = region.view(nb_piece, unit)
        codes = region[:, :unit - 1].contiguous().view(torch.int8).view(nb_piece, block)
        codec = quant_meta.get("codec") or "int8"
        if codec == "int8":
            payload = codes.float()
        elif codec == "fp8_e4m3":
            payload = codes.view(torch.float8_e4m3fn).float()
        else:
            raise ValueError(f"unsupported quant codec {codec!r}")
        scale = region[:, unit - 1].unsqueeze(1)
        deq = (payload * scale).view(-1)[: pe - ps]
        if sqrt_domain:
            deq = deq.square()  # 保存端在 sqrt 域量化，恢复端平方还原
        dst_flat.copy_(deq)

    def _resolve_published_slot(self):
        """返回 (slot, counter, source)。新格式文件读已发布记录；从未发布则报错；
        旧格式文件回退 metadata 的 latest_parall_iter（崩溃一致性无保证，打警告）。"""
        rec = _ms_format().read_published_checkpoint(self.checkpoint_file, self.max_async)
        if rec["status"] == "published":
            hint = self.metadata.get("latest_parall_iter")
            if hint is not None and int(hint) != rec["slot"]:
                print(f"[WARN] metadata latest_parall_iter={hint} disagrees with published "
                      f"record slot={rec['slot']}; using the published record")
            return rec["slot"], rec["counter"], "published"
        if rec["status"] == "none":
            raise RuntimeError(
                f"{self.checkpoint_file}: no checkpoint has been published yet "
                f"(registered slots may hold partial data)")
        hint = self.metadata.get("latest_parall_iter")
        print(f"[WARN] {self.checkpoint_file}: legacy file without published record; "
              f"falling back to metadata latest_parall_iter={hint} (not crash-consistent)")
        return int(hint if hint is not None else 0), None, "legacy-metadata"

    @property
    def published_payload(self) -> Optional[Dict[str, Any]]:
        """该 slot extra blob 中由保存端写入的发布载荷（如 PP 的 completed_steps）。"""
        st = getattr(self, "_extra_state_early", None) or {}
        return st.get("publish_payload")

    # ================== 非分层状态（extra state, CSXS blob） ==================
    def load_extra_state(self) -> Optional[Dict[str, Any]]:
        """读取并校验本 slot 的非分层状态 blob（RNG/scheduler/scaler/用户状态）。

        校验链：magic → 长度 → crc32（parse_extra_state_blob 内）→
        blob.parall_iter == 本 loader 的 slot（不匹配说明 slot 已被更新的
        checkpoint 复用，metadata 过期）。
        """
        if not self.extra_state_enabled or self.extra_stream_floats <= 0:
            return None
        buf = torch.empty(self.extra_stream_floats, dtype=torch.float32)
        self.lib.read_stream_chunk(
            self.reader_obj, int(self.extra_stream_id), c_void_p(buf.data_ptr()),
            c_size_t(0), c_size_t(self.extra_stream_floats), c_int(self.parall_iter))
        raw = buf.numpy().tobytes()
        ms = _ms_format()
        state, _ = ms.MultiStreamCheckpoint.parse_extra_state_blob(raw)
        blob_slot = int(state.get("parall_iter", -1))
        if blob_slot != int(self.parall_iter):
            raise ValueError(
                f"extra-state blob parall_iter={blob_slot} != loader slot "
                f"{self.parall_iter}; slot may have been reused by a newer "
                f"checkpoint (stale latest_parall_iter in metadata?)")
        return state

    @staticmethod
    def apply_extra_state(state: Dict[str, Any], scheduler=None, scaler=None,
                          set_rng: bool = True) -> Optional[Dict[str, Any]]:
        """应用非分层状态；返回应用自定义的 user dict（如 dataloader 位置）。"""
        rng = state.get("rng") or {}
        if set_rng and rng:
            if rng.get("torch_cpu") is not None:
                torch.set_rng_state(rng["torch_cpu"])
            cuda_states = rng.get("torch_cuda")
            if cuda_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_states)
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("python") is not None:
                random.setstate(rng["python"])
        if scheduler is not None and "lr_scheduler" in state:
            scheduler.load_state_dict(state["lr_scheduler"])
        if scaler is not None and "grad_scaler" in state:
            scaler.load_state_dict(state["grad_scaler"])
        return state.get("user")

    def stop(self) -> None:
        self.producer_stop_event.set()
        self.copy_worker_stop_event.set()
        for _ in range(self.num_copy_workers + 2):
            try:
                self.host_buffer_queue.put_nowait(None)
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

    def _wait_phase(self, chunk_idx: int, cpu_events, gpu_events) -> None:
        self._raise_if_fatal_error()
        if 0 <= chunk_idx < self.num_chunks:
            # 先等 CPU 侧就绪信号（保证 CUDA event 已 record——未 record 的
            # event 上 wait_event 是空操作），再做 GPU 侧顺序约束。
            while not cpu_events[chunk_idx].wait(timeout=1.0):
                self._raise_if_fatal_error()
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.current_stream().wait_event(gpu_events[chunk_idx])  # type: ignore[arg-type]
        self._raise_if_fatal_error()

    def wait_for_chunk(self, chunk_idx: int) -> None:
        """等待 chunk 的 param 流就绪（forward pre-hook 语义）。"""
        self._wait_phase(chunk_idx, self._param_done_cpu, self.param_events)

    def wait_for_chunk_full(self, chunk_idx: int) -> None:
        """等待 chunk 的全部状态流（moments/master）就绪。"""
        self._wait_phase(chunk_idx, self._full_done_cpu, self.full_events)

    def wait_all_full(self) -> None:
        """阻塞到所有 chunk 的全部状态流恢复完成——首个 optimizer.step 前的屏障。"""
        for i in range(self.num_chunks):
            self.wait_for_chunk_full(i)

    def is_chunk_loaded(self, chunk_idx: int) -> bool:
        """param 流是否就绪（worker 在 stream.synchronize 后置位 ⇒ 数据已上卡）。"""
        self._raise_if_fatal_error()
        if 0 <= chunk_idx < self.num_chunks:
            return bool(self._chunk_param_loaded[chunk_idx])
        return False

    def is_chunk_fully_loaded(self, chunk_idx: int) -> bool:
        self._raise_if_fatal_error()
        if 0 <= chunk_idx < self.num_chunks:
            return bool(self._chunk_full_loaded[chunk_idx])
        return False

    # ============ distributed.py 兼容薄层（多卡 PP 恢复 API，源自 distributed 分支） ============
    # ForwardLoadGate / PipelayerAsyncLoadHandle 依赖以下命名。
    # 语义映射：param 级就绪 = forward 可跑（weights-first phase A）；
    #          full 级就绪 = optimizer 可更新（phase B，moments/master 全部上卡）。

    def is_forward_group_ready(self, chunk_idx: int) -> bool:
        return self.is_chunk_loaded(chunk_idx)

    def wait_for_forward_group(self, chunk_idx: int) -> None:
        self.wait_for_chunk(chunk_idx)

    def is_forward_ready(self) -> bool:
        return all(self.is_chunk_loaded(i) for i in range(self.num_chunks))

    def wait_forward_ready(self) -> None:
        for i in range(self.num_chunks):
            self.wait_for_chunk(i)

    def is_update_ready(self) -> bool:
        return all(self.is_chunk_fully_loaded(i) for i in range(self.num_chunks))

    def wait_update_ready(self) -> None:
        self.wait_all_full()

    def get_restore_stats(self) -> Dict[str, Any]:
        return {
            "num_chunks": int(self.num_chunks),
            "param_chunks_loaded": sum(
                1 for i in range(self.num_chunks) if self.is_chunk_loaded(i)),
            "full_chunks_loaded": sum(
                1 for i in range(self.num_chunks) if self.is_chunk_fully_loaded(i)),
            "weights_first": bool(self.weights_first),
        }

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
