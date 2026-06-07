#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper 字幕提取服务
负责从视频中提取字幕
"""

import os
import threading
import warnings
from pathlib import Path
from typing import Optional, Callable
from faster_whisper import WhisperModel

from core.models import WhisperConfig, VADParameters
from utils.format_utils import format_timestamp, format_file_size


# CUDA/cuBLAS 相关错误的特征子串，用于判断是否需要回退到 CPU
_CUDA_ERROR_MARKERS = ("libcublas", "cuda", "cudnn")

# Whisper 模型大小 → HuggingFace 仓库名映射
_MODEL_REPO_MAP = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# Whisper 模型大小 → 完整下载字节数（从 huggingface.co/api/models/.../tree/main 拉取）
# 用于下载进度计算（百分比 / 总大小显示）。
# 数字是 2026-06 拉的真实大小，可能随模型版本小幅波动。
MODEL_TOTAL_SIZE_BYTES = {
    "tiny":     78_207_087,    #  74.6 MB
    "base":    147_886_409,    # 141.0 MB
    "small":   486_215_847,    # 463.7 MB
    "medium": 1_530_575_217,   #   1.5 GB
    "large-v3": 3_090_839_273, #   2.9 GB
}


def _calc_download_pct(current_bytes: int, model_size: str) -> float:
    """
    计算 Whisper 模型下载的段内进度（0.0 - 100.0）

    v1.8.1: 返回两位小数 float，clamp 到 [0, 100]。

    旧版本（v1.7.7-v1.8.0）返回 int，clamp 到 [5, 99]，避免 0% 让用户以为卡死
    也不显示 100%（避免和后续字幕提取阶段的 50% 衔接时 UI 回跳）。
    新版本不再需要这些 hack：下载是独立阶段，0% 就是 0%，100% 就是 100%。

    Args:
        current_bytes: 已下载字节数
        model_size: 模型大小 ('tiny', 'base', 'small', 'medium', 'large-v3')

    Returns:
        0.0 - 100.0 的两位小数
    """
    total = MODEL_TOTAL_SIZE_BYTES.get(model_size, 0)
    if total <= 0 or current_bytes <= 0:
        return 0.0
    raw = (current_bytes / total) * 100
    return round(min(100.0, raw), 2)


def _format_download_message(current_bytes: int, model_size: str) -> str:
    """
    格式化下载进度消息

    Args:
        current_bytes: 已下载字节数
        model_size: 模型大小

    Returns:
        "正在下载模型 X... 47.32 MB / 74.6 MB (47.32%)" 或 fallback
    """
    from utils.format_utils import format_file_size

    total = MODEL_TOTAL_SIZE_BYTES.get(model_size, 0)
    if total > 0 and current_bytes > 0:
        pct = _calc_download_pct(current_bytes, model_size)
        return (
            f"正在下载模型 {model_size}... "
            f"{format_file_size(current_bytes)} / "
            f"{format_file_size(total)} ({pct:.2f}%)"
        )
    elif current_bytes > 0:
        return (
            f"正在下载模型 {model_size}... "
            f"已下载 {format_file_size(current_bytes)}"
        )
    else:
        return f"正在下载模型 {model_size}..."


def _check_hf_endpoint_compat() -> None:
    """
    检测并警告 HF_ENDPOINT 与 huggingface_hub 不兼容的常见配置。

    历史背景：
      huggingface_hub >= 1.18 的 _httpx_follow_relative_redirects_with_backoff
      不再 follow 跨域 308 重定向，而 hf-mirror.com 正是用 308 → 307 → huggingface.co
      链路工作。设 HF_ENDPOINT=https://hf-mirror.com 时，HEAD 请求拿到 308 响应后
      因无 x-repo-commit header 而抛 FileMetadataError，表现为：
        "An error happened while trying to locate the file on the Hub..."

    建议：直连 huggingface.co，不要设 HF_ENDPOINT。
    """
    hf_endpoint = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if not hf_endpoint:
        return  # 未设置，无需处理

    # 已知不兼容的镜像源
    bad_endpoints = ("hf-mirror.com",)
    if any(bad in hf_endpoint for bad in bad_endpoints):
        warnings.warn(
            f"[WhisperService] 检测到 HF_ENDPOINT={hf_endpoint!r}，"
            f"该镜像源与 huggingface_hub >= 1.18 不兼容（不 follow 跨域 308 重定向），"
            f"会导致 'locate the file on the Hub' 错误。"
            f"建议清空 HF_ENDPOINT 环境变量后重试。",
            RuntimeWarning,
            stacklevel=2,
        )
        print(
            f"[WhisperService] ⚠️  HF_ENDPOINT={hf_endpoint} 与新版 huggingface_hub 不兼容，"
            f"将尝试自动 unset 后再下载"
        )
        # 自动清除，避免触发已知 bug
        os.environ.pop("HF_ENDPOINT", None)
        os.environ.pop("hf_endpoint", None)


def get_model_dir() -> str:
    """获取模型存储目录（Docker 用 /data/models，本地开发用 ./data/models）"""
    docker_path = "/data/models"
    if os.path.isdir(docker_path):
        return docker_path
    return "./data/models"


def is_model_downloaded(model_size: str, model_dir: str = None) -> bool:
    """
    检查指定 Whisper 模型是否已下载到本地。

    通过检查 HF 缓存目录结构判断：
    {model_dir}/models--Systran--faster-whisper-{size}/snapshots/ 下有文件即为已下载。

    ⚠️ 2026-06-06 改进：增加完整性校验。
    之前只检查 snapshots/ 下是否有非空目录，但半下载状态会创建空目录 +
    .incomplete 文件，导致 _is_model_cached() 假阳性返回 True，
    WhisperModel 加载时尝试加载半下载文件 → 卡住/报错。
    现在的判断：必须满足"目录非空" + "所有 required_files 都存在且 size > 0"。

    Args:
        model_size: 模型大小，如 "tiny", "base", "small", "medium", "large-v3"
        model_dir: 模型目录，默认自动检测
    Returns:
        True 表示已下载且完整
    """
    if model_dir is None:
        model_dir = get_model_dir()
    repo_name = f"models--Systran--faster-whisper-{model_size}"
    snapshots = Path(model_dir) / repo_name / "snapshots"
    if not snapshots.is_dir():
        return False
    # 2026-06-06 改进：必须满足 _verify_model_files 才算下载完成
    required_files = ["config.json", "model.bin", "tokenizer.json"]
    for commit_dir in snapshots.iterdir():
        if not commit_dir.is_dir():
            continue
        if all((commit_dir / f).exists() and (commit_dir / f).stat().st_size > 0
               for f in required_files):
            return True
    return False


class WhisperService:
    """Whisper 字幕提取服务"""
    
    def __init__(
        self,
        config: WhisperConfig,
        vad_params: VADParameters,
        model_dir: str = "/data/models"
    ):
        """
        初始化 Whisper 服务
        
        Args:
            config: Whisper 配置
            vad_params: VAD 参数
            model_dir: 模型存储目录
        """
        self.config = config
        self.vad_params = vad_params
        self.model_dir = model_dir
        self.model: Optional[WhisperModel] = None
    
    def _is_model_cached(self) -> bool:
        """
        检查模型文件是否已完整下载到本地缓存。

        2026-06-06 改进：之前只检查 snapshots 目录下是否有非空 commit 目录,
        但半下载状态（.incomplete 文件 / 部分下载）也会创建非空目录，
        导致假阳性返回 True，WhisperModel 加载时尝试加载半下载文件 → 卡住。
        现在直接复用 _verify_model_files() 的逻辑：必须所有关键文件存在且 size > 0。
        """
        return self._verify_model_files()

    def _verify_model_files(self) -> bool:
        """验证模型关键文件是否完整（检测半下载状态）"""
        required_files = ["config.json", "model.bin", "tokenizer.json"]
        model_dir = Path(self.model_dir)
        repo_name = f"models--Systran--faster-whisper-{self.config.model_size}"
        snapshots = model_dir / repo_name / "snapshots"
        if not snapshots.is_dir():
            return False
        for commit_dir in snapshots.iterdir():
            if not commit_dir.is_dir():
                continue
            if all((commit_dir / f).exists() and (commit_dir / f).stat().st_size > 0
                   for f in required_files):
                return True
        return False

    def _resolve_device(self) -> str:
        """解析实际使用的 device：WHISPER_DEVICE 环境变量优先于 config.whisper.device。"""
        env_device = os.environ.get("WHISPER_DEVICE")
        if env_device and env_device.strip():
            return env_device.strip()
        return self.config.device

    @staticmethod
    def _has_cuda_device() -> bool:
        """检查容器内是否真的能识别到 CUDA 设备"""
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    @staticmethod
    def _is_cuda_error(exc: Exception) -> bool:
        """判断异常是否与 CUDA/cuBLAS 库加载失败相关（用于决定是否回退 CPU）。"""
        msg = str(exc).lower()
        return any(marker in msg for marker in _CUDA_ERROR_MARKERS)

    def load_model(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ):
        """
        加载 Whisper 模型

        行为：
        - device="auto"：先预检查 CUDA 可用性，没有则直接用 CPU；
                        有则先尝试 cuda，失败回退 cpu
        - device="cuda" / "cpu"：显式指定，不静默回退，错误原样抛出
        - WHISPER_DEVICE 环境变量覆盖 config.whisper.device
        - 模型未下载或文件不完整时显示下载进度

        Args:
            progress_callback: 进度回调 (current, total, message)，用于在下载时上报进度
        """
        if self.model is not None:
            return

        # 在加载前检查 HF_ENDPOINT 与新版 huggingface_hub 的兼容性
        # 若发现 hf-mirror.com 等不兼容源，自动清除并警告
        _check_hf_endpoint_compat()

        # 完整性检测：发现半下载状态时视为未缓存，触发重新下载
        if self._is_model_cached() and not self._verify_model_files():
            print(
                f"[WhisperService] 模型文件不完整，将重新下载: "
                f"{self.config.model_size}"
            )
            is_cached = False
        else:
            is_cached = self._is_model_cached()

        if not is_cached and progress_callback:
            model_total_bytes = MODEL_TOTAL_SIZE_BYTES.get(
                self.config.model_size, 0
            )
            # 启动消息：含总大小预期（让用户心里有数）
            if model_total_bytes > 0:
                initial_msg = (
                    f"首次使用，正在下载模型 {self.config.model_size}... "
                    f"总大小 {format_file_size(model_total_bytes)}"
                )
            else:
                initial_msg = (
                    f"首次使用，正在下载模型 {self.config.model_size}..."
                )
            progress_callback("download", 0.0, initial_msg)

            # 启动后台线程轮询下载目录大小，定时上报进度
            stop_event = threading.Event()
            cancel_event = threading.Event()  # 2026-06-06 新增: 取消信号
            model_dir = Path(self.model_dir)

            def _poll_download():
                # v1.8.1: 下载阶段独立进度 0-100% (两位小数)，
                # 不再走旧的 [5, 99] 区间 hack，UI 会按 stage 字段展示。
                while not stop_event.is_set():
                    try:
                        if model_total_bytes > 0:
                            current_bytes = sum(
                                f.stat().st_size
                                for f in model_dir.rglob('*')
                                if f.is_file()
                            )
                            pct = _calc_download_pct(
                                current_bytes, self.config.model_size
                            )
                            progress_callback(
                                "download", pct,
                                _format_download_message(
                                    current_bytes, self.config.model_size
                                )
                            )
                        else:
                            current_bytes = sum(
                                f.stat().st_size
                                for f in model_dir.rglob('*')
                                if f.is_file()
                            )
                            if current_bytes > 0:
                                progress_callback(
                                    "download", 0.0,
                                    f"正在下载模型 {self.config.model_size}... "
                                    f"已下载 {format_file_size(current_bytes)}"
                                )
                    except Exception:
                        # progress_callback 内部可能 raise InterruptedError
                        # (worker 检测到取消),这里捕获后设置 cancel_event
                        # 让外层 WhisperModel 调用也能退出
                        cancel_event.set()
                        stop_event.set()
                        return
                    # 用更短的 wait (1s) 让取消信号更快被检测到
                    if stop_event.wait(1):
                        break

            poll_thread = threading.Thread(target=_poll_download, daemon=True)
            poll_thread.start()
        else:
            stop_event = None

        # 决定本次加载的 device：env 优先；'auto' 表示先试 cuda 再回退 cpu
        device = self._resolve_device()
        compute_type = self.config.compute_type

        def _do_load(dev: str, ct: str):
            return WhisperModel(
                self.config.model_size,
                device=dev,
                compute_type=ct,
                download_root=self.model_dir
            )

        try:
            if device == "auto":
                # auto：先预检查 CUDA 可用性，没有则直接走 CPU
                if not self._has_cuda_device():
                    print(
                        f"[WhisperService] 容器内未检测到 CUDA 设备，直接使用 CPU"
                    )
                    self.model = _do_load("cpu", compute_type)
                    print(
                        f"[WhisperService] Model loaded: "
                        f"{self.config.model_size} (device=cpu)"
                    )
                else:
                    # 有 CUDA：先尝试 cuda，失败（CUDA/cuBLAS 不可用）回退 cpu
                    try:
                        self.model = _do_load("cuda", compute_type)
                        print(
                            f"[WhisperService] Model loaded: "
                            f"{self.config.model_size} (device=cuda)"
                        )
                    except Exception as e:
                        if self._is_cuda_error(e):
                            print(
                                f"[WhisperService] CUDA 不可用 ({e})，回退到 CPU"
                            )
                            self.model = _do_load("cpu", compute_type)
                            print(
                                f"[WhisperService] Model loaded: "
                                f"{self.config.model_size} (device=cpu)"
                            )
                        else:
                            raise
            else:
                # 显式 cuda / cpu：不静默回退，错误原样向上抛
                self.model = _do_load(device, compute_type)
                print(f"[WhisperService] Model loaded: {self.config.model_size}")
        except Exception as e:
            print(f"[WhisperService] Failed to load model: {e}")
            raise
        finally:
            if stop_event:
                stop_event.set()

    def extract_subtitle(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> str:
        """
        从视频中提取字幕
        
        Args:
            video_path: 视频文件路径
            output_path: 输出 SRT 文件路径（默认：同名 .srt）
            progress_callback: 进度回调函数 (current, total, message)
        
        Returns:
            生成的 SRT 文件路径
        """
        # 确保模型已加载（传入回调以支持下载进度上报）
        if self.model is None:
            self.load_model(progress_callback)
        
        # 确定输出路径
        if output_path is None:
            output_path = str(Path(video_path).with_suffix('.srt'))
        
        # 更新进度（v1.8.1: 新协议 stage='extract'，stage_progress 独立 0-100%）
        if progress_callback:
            progress_callback("extract", 0.0, f"开始提取字幕...")
        
        # 准备转录参数
        transcribe_params = {
            'audio': video_path,
            'beam_size': 5,
            'vad_filter': True,
            'vad_parameters': self.vad_params.to_dict(),
            'word_timestamps': True,
            'condition_on_previous_text': True,
            'temperature': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
        
        # 如果不是自动检测，指定语言
        if self.config.source_language != 'auto':
            transcribe_params['language'] = self.config.source_language
        
        try:
            # 执行转录
            segments, info = self.model.transcribe(**transcribe_params)
            
            # 更新进度
            if progress_callback:
                from utils.format_utils import get_lang_name
                lang_name = get_lang_name(info.language)
                progress_callback("extract", 5.0, f"检测语言: {lang_name}")
            
            # 写入 SRT 文件
            with open(output_path, 'w', encoding='utf-8') as f:
                idx = 0
                for seg in segments:
                    idx += 1
                    
                    # 写入字幕条目
                    f.write(f"{idx}\n")
                    f.write(
                        f"{format_timestamp(seg.start)} --> "
                        f"{format_timestamp(seg.end)}\n"
                    )
                    f.write(f"{seg.text.strip()}\n\n")
                    
                    # 更新进度
                    if progress_callback and idx % 10 == 0:
                        # v1.8.1: 提取阶段 5-95% 区间（5% 给"检测语言"，95% 给"完成"）
                        # 不再硬塞进 5-50% 区间（worker 不再算 50+int(...)）
                        progress = 5.0 + min(90.0, (idx / 300) * 90.0)
                        progress_callback("extract", progress, f"已转写 {idx} 行")
            
            # 完成
            if progress_callback:
                progress_callback("extract", 100.0, f"字幕提取完成 ({idx} 行)")
            
            return output_path
        
        except Exception as e:
            print(f"[WhisperService] Extraction failed: {e}")
            raise
    
    def unload_model(self):
        """卸载模型（释放内存）"""
        if self.model is not None:
            del self.model
            self.model = None
            print("[WhisperService] Model unloaded")


# ============================================================================
# 快捷函数
# ============================================================================

def extract_subtitle_from_video(
    video_path: str,
    config: WhisperConfig,
    vad_params: VADParameters,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> str:
    """
    从视频提取字幕（快捷函数）
    
    Args:
        video_path: 视频文件路径
        config: Whisper 配置
        vad_params: VAD 参数
        output_path: 输出路径（可选）
        progress_callback: 进度回调
    
    Returns:
        SRT 文件路径
    """
    service = WhisperService(config, vad_params)
    return service.extract_subtitle(video_path, output_path, progress_callback)
