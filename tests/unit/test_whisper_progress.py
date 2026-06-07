"""Tests for Whisper download progress callback format (v1.7.7 fix).

背景：之前 load_model() 里的 _poll_download() 把 progress_callback
的 current 写死成 5,用户只看到"已下载 X MB",看不到百分比和总大小。

修复后应该：
- callback current 是真实百分比 (5-99)
- message 同时显示 current_bytes / total_bytes (XX%)
- 如果有未知 model size 走 fallback（只有已下载大小）

注意：这些测试是纯常量/格式守卫，不依赖 faster_whisper，
可以在本地（缺 faster_whisper 的环境）跑。
"""

import pytest


# ---------------------------------------------------------------------------
# 常量守卫
# ---------------------------------------------------------------------------

class TestModelTotalSizeBytes:
    """MODEL_TOTAL_SIZE_BYTES 必须覆盖所有有效 Whisper 模型大小"""

    def test_contains_all_valid_sizes(self):
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES, _MODEL_REPO_MAP
        # VALID_SIZES 跟 _MODEL_REPO_MAP 一样:
        # tiny, base, small, medium, large-v3
        for size in _MODEL_REPO_MAP.keys():
            assert size in MODEL_TOTAL_SIZE_BYTES, (
                f"MODEL_TOTAL_SIZE_BYTES 缺少 {size!r}，"
                f"下载进度无法计算百分比"
            )

    @pytest.mark.parametrize("size", ["tiny", "base", "small", "medium", "large-v3"])
    def test_size_is_positive_integer(self, size):
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES
        size_bytes = MODEL_TOTAL_SIZE_BYTES[size]
        assert isinstance(size_bytes, int), f"{size} size 必须是 int"
        assert size_bytes > 0, f"{size} size 必须 > 0"

    @pytest.mark.parametrize("size", ["tiny", "base", "small", "medium", "large-v3"])
    def test_size_reasonable_range(self, size):
        """模型大小应该在 50MB - 5GB 之间（粗略 sanity check）"""
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES
        size_bytes = MODEL_TOTAL_SIZE_BYTES[size]
        # tiny 至少 50MB，large-v3 不超过 5GB
        assert 50 * 1024 * 1024 <= size_bytes <= 5 * 1024 * 1024 * 1024, (
            f"{size} size {size_bytes} 超出合理范围 [50MB, 5GB]"
        )

    def test_sizes_increase(self):
        """tiny < base < small < medium < large-v3"""
        from services.whisper_service import MODEL_TOTAL_SIZE_BYTES
        sizes = [MODEL_TOTAL_SIZE_BYTES[k] for k in
                 ["tiny", "base", "small", "medium", "large-v3"]]
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1], (
                f"模型大小应该递增: {sizes}"
            )


# ---------------------------------------------------------------------------
# 进度计算函数（纯逻辑，提取出来方便单测）
# ---------------------------------------------------------------------------

class TestProgressCalculation:
    """下载进度的百分比计算 + message 格式

    实际逻辑在 whisper_service.load_model._poll_download() 内嵌函数里,
    这里把核心算法抽出来单测,防止 regression。
    """

    @pytest.fixture
    def calc(self):
        """v1.8.1: 直接用 whisper_service 的真函数（避免 fixture 跟真代码漂移）"""
        from services.whisper_service import (
            _calc_download_pct,
            _format_download_message,
        )
        return _calc_download_pct, _format_download_message

    @pytest.mark.parametrize("model_size,expected_total", [
        ("tiny", 78_207_087),
        ("base", 147_886_409),
        ("small", 486_215_847),
        ("medium", 1_530_575_217),
        ("large-v3", 3_090_839_273),
    ])
    def test_pct_at_zero_is_zero(self, calc, model_size, expected_total):
        """0 字节时进度是 0.00%（v1.8.1: 不再 clamp 到 5%，0 就是 0）"""
        _calc_pct, _ = calc
        assert _calc_pct(0, model_size) == 0.0

    @pytest.mark.parametrize("model_size,expected_total", [
        ("tiny", 78_207_087),
        ("base", 147_886_409),
        ("medium", 1_530_575_217),
    ])
    def test_pct_at_half(self, calc, model_size, expected_total):
        """下载一半应该是 50.00%"""
        _calc_pct, _ = calc
        assert _calc_pct(expected_total // 2, model_size) == 50.0

    def test_pct_at_100(self, calc):
        """下载完成应该是 100.00%"""
        _calc_pct, _ = calc
        assert _calc_pct(78_207_087, "tiny") == 100.0

    def test_pct_above_100_is_clamped(self, calc):
        """超过总大小应该 clamp 到 100（v1.8.1 改：不再 99）"""
        _calc_pct, _ = calc
        assert _calc_pct(78_207_087 * 2, "tiny") == 100.0

    def test_pct_returns_float_with_two_decimals_precision(self, calc):
        """47.32% 这样的两位小数能精确表示（round-trip）"""
        _calc_pct, _ = calc
        # 35% of tiny
        target_bytes = int(78_207_087 * 0.35)
        pct = _calc_pct(target_bytes, "tiny")
        # 允许 0.01 误差（round 之后的浮点）
        assert abs(pct - 35.0) < 0.01

    def test_pct_unknown_model_returns_zero(self, calc):
        """未知 model size 返回 0.0（不再 fallback 到 5）"""
        _calc_pct, _ = calc
        assert _calc_pct(1_000_000, "nonexistent-model") == 0.0

    def test_message_contains_all_three_pieces(self, calc):
        """message 必须同时显示 已下载 / 总大小 / 百分比 三项"""
        _, _format_message = calc
        msg = _format_message(78_207_087 // 2, "tiny")  # 50% tiny
        # 总大小（74.6 MB，固定）
        assert "74.6 MB" in msg, f"缺总大小: {msg}"
        # 百分比：v1.8.1 改为两位小数
        assert "(50.00%)" in msg, f"缺两位小数百分比: {msg}"
        # 已下载：format_file_size 用 1024 进制，78_207_087/2/1024/1024 = 37.3 MB
        # 不强求确切值，只验证有"MB" + 数字格式
        import re
        assert re.search(r"\d+\.\d+ MB / 74\.6 MB", msg), (
            f"已下载大小格式不对: {msg}"
        )

    def test_message_uses_full_path(self, calc):
        """message 包含模型名（不是只显示"下载"）"""
        _, _format_message = calc
        msg = _format_message(100_000_000, "medium")
        assert "medium" in msg
        # 进度数字
        assert "%" in msg

    def test_message_uses_two_decimal_format(self, calc):
        """v1.8.1: 百分比必须是两位小数格式 (.2f)"""
        _, _format_message = calc
        # 任意非零进度都应该是 XX.XX% 格式
        msg = _format_message(78_207_087 // 3, "tiny")
        import re
        # 匹配 "(数字.数字%)" 形式
        assert re.search(r"\(\d+\.\d{2}%\)", msg), (
            f"百分比不是两位小数: {msg}"
        )

    def test_message_fallback_when_unknown_size(self, calc):
        """未知 model size 只显示"已下载"（兼容老逻辑）"""
        _, _format_message = calc
        msg = _format_message(1_000_000, "nonexistent")
        assert "已下载" in msg
        # 不能有误导性的百分比
        assert "(%)" not in msg
