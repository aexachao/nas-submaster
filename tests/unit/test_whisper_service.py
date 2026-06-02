"""Tests for WhisperService GPU detection and CPU fallback (issue #4).

覆盖场景：
- device='auto' + libcublas 错误 → 自动回退 CPU int8
- device='cuda' + libcublas 错误 → 显式 cuda 不静默回退，抛错
- device='auto' + 非 CUDA 错误 → 不回退，抛错
- device='cpu' → 跳过 GPU 尝试
- WHISPER_DEVICE 环境变量 → 覆盖 config.whisper.device
- load_model 幂等性
"""
from unittest.mock import MagicMock

import pytest

from core.models import VADParameters, WhisperConfig
from services.whisper_service import WhisperService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vad_params() -> VADParameters:
    return VADParameters(
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=2000,
        speech_pad_ms=400,
    )


@pytest.fixture
def cached_model_dir(tmp_path) -> str:
    """创建一个看起来已缓存的模型目录，避免 load_model 启动下载轮询线程。"""
    model_dir = tmp_path / "models"
    cache_subdir = model_dir / "models--Systran--faster-whisper-tiny"
    cache_subdir.mkdir(parents=True)
    (cache_subdir / "marker.txt").write_text("fake model file for test")
    return str(model_dir)


@pytest.fixture
def make_service(vad_params, cached_model_dir):
    def _make(device="cpu", compute_type="int8", model_size="tiny") -> WhisperService:
        config = WhisperConfig(
            model_size=model_size,
            compute_type=compute_type,
            device=device,
            source_language="auto",
        )
        return WhisperService(config, vad_params, model_dir=cached_model_dir)
    return _make


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_auto_falls_back_to_cpu_on_libcublas_error(
    make_service, mocker, monkeypatch
):
    """device='auto' 时遇到 libcublas 错误应自动回退到 CPU。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="auto")

    mock_model = MagicMock(name="WhisperModelInstance")
    calls: list[dict] = []

    def side_effect(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "Library libcublas.so.12 is not found or cannot be loaded"
            )
        return mock_model

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    service.load_model()

    assert len(calls) == 2, f"Expected cuda + cpu fallback, got {len(calls)} call(s)"
    # 第一次尝试用 cuda
    assert calls[0]["device"] == "cuda"
    # 回退到 cpu
    assert calls[1]["device"] == "cpu"
    assert service.model is mock_model


def test_explicit_cuda_does_not_silently_fallback(
    make_service, mocker, monkeypatch
):
    """device='cuda' 显式指定时，libcublas 错误不应被静默吞掉回退。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="cuda")

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError(
            "Library libcublas.so.12 is not found or cannot be loaded"
        )

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    with pytest.raises(RuntimeError, match="libcublas"):
        service.load_model()
    # 不应回退：只调用一次
    assert call_count["n"] == 1


def test_auto_does_not_swallow_non_cuda_errors(
    make_service, mocker, monkeypatch
):
    """device='auto' 模式下，非 libcublas/CUDA 错误应直接抛出，不回退。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="auto")

    call_count = {"n": 0}

    def side_effect(*args, **kwargs):
        call_count["n"] += 1
        raise RuntimeError("Some other error: out of memory")

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        service.load_model()
    # 错误与 CUDA 无关，不应触发回退
    assert call_count["n"] == 1


def test_cpu_device_skips_gpu_attempt(make_service, mocker, monkeypatch):
    """device='cpu' 时直接走 CPU，不应尝试 CUDA。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="cpu", compute_type="int8")

    mock_model = MagicMock(name="WhisperModelInstance")
    captured: list[dict] = []

    def side_effect(*args, **kwargs):
        captured.append(kwargs)
        return mock_model

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    service.load_model()

    assert len(captured) == 1
    assert captured[0]["device"] == "cpu"
    assert captured[0]["compute_type"] == "int8"


def test_env_var_overrides_config_device(make_service, mocker, monkeypatch):
    """WHISPER_DEVICE 环境变量应覆盖 config.whisper.device。"""
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    service = make_service(device="cpu")  # config 写 cpu，env 覆盖为 cuda

    mock_model = MagicMock(name="WhisperModelInstance")
    captured: list[dict] = []

    def side_effect(*args, **kwargs):
        captured.append(kwargs)
        return mock_model

    mocker.patch(
        "services.whisper_service.WhisperModel", side_effect=side_effect
    )

    service.load_model()

    assert len(captured) == 1
    assert captured[0]["device"] == "cuda"


def test_load_model_is_idempotent(make_service, mocker, monkeypatch):
    """重复调用 load_model 不应重复构造模型。"""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    service = make_service(device="cpu")

    mock_model = MagicMock(name="WhisperModelInstance")
    mock_class = mocker.patch(
        "services.whisper_service.WhisperModel", return_value=mock_model
    )

    service.load_model()
    service.load_model()

    assert mock_class.call_count == 1
    assert service.model is mock_model
