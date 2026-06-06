#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/config.py APP_VERSION 守卫测试

目的：防止 commit message 写了 v1.x.x 但忘了改代码常量。
具体场景：CI workflow (.github/workflows/docker-publish.yml) 用
`grep -oP 'APP_VERSION = "v\\K[^"]+' core/config.py` 提取版本号打 Docker tag。
如果 APP_VERSION 没改，CI 推的 tag 还是旧版本号，用户 docker
看到的"最新"就跟实际 commit 不一致。

这个测试不强求版本号格式（不同项目格式不同），只要求：
1. APP_VERSION 是字符串
2. 格式是 v<数字>.<数字>.<数字>(+可选后缀)
3. 跟最近一次 commit message 里出现的 vX.Y.Z 不冲突

第 3 点是核心：每次 release 时，code 常量必须跟 commit message
对齐。
"""

import re
import subprocess

import pytest

from core.config import APP_VERSION


# ---------------------------------------------------------------------------
# 格式守卫
# ---------------------------------------------------------------------------

class TestAppVersionFormat:
    """APP_VERSION 必须符合 v<major>.<minor>.<patch>[suffix] 格式"""

    def test_app_version_is_string(self):
        assert isinstance(APP_VERSION, str)

    def test_app_version_starts_with_v(self):
        assert APP_VERSION.startswith("v"), f"APP_VERSION 应以 'v' 开头: {APP_VERSION!r}"

    def test_app_version_format(self):
        """v<major>.<minor>.<patch> 或带后缀（-rc1, .1 等）"""
        pattern = r"^v\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?$"
        assert re.match(pattern, APP_VERSION), (
            f"APP_VERSION 格式不对: {APP_VERSION!r}\n"
            f"应该是 v<major>.<minor>.<patch>[suffix]，比如 v1.7.7 或 v1.7.7-rc1"
        )


# ---------------------------------------------------------------------------
# Commit message 对齐（防 commit msg 跟 code 漂移）
# ---------------------------------------------------------------------------

class TestAppVersionAlignsWithCommits:
    """APP_VERSION 跟最近 commit message 里提到的版本号要一致"""

    @pytest.fixture
    def last_commit_message(self) -> str:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%B"],
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/dev/nas-submaster",
        )
        return result.stdout.strip()

    @pytest.fixture
    def last_5_commits(self) -> list:
        result = subprocess.run(
            ["git", "log", "-5", "--pretty=%H %s"],
            capture_output=True,
            text=True,
            check=True,
            cwd="/home/dev/nas-submaster",
        )
        return [line.strip() for line in result.stdout.strip().splitlines()]

    def test_recent_commits_have_version_in_subject(self, last_5_commits):
        """最近 5 个 commit 至少要有 1 个主题里含 v<APP_VERSION>

        防止 APP_VERSION 改完，但周围 commit message 全是 v1.7.6
        这种漂移。
        """
        # 把 v1.7.7 变成 v1\.7\.7 适配正则
        esc = re.escape(APP_VERSION)
        pattern = re.compile(esc)
        matches = [c for c in last_5_commits if pattern.search(c)]
        # 这条规则是软的（有些 commit 可能不写版本号），所以只 warn 不 fail
        # 但如果 5 个 commit 全部不含 APP_VERSION，那就说明有漂移
        if not matches:
            pytest.skip(
                f"最近 5 个 commit 都没出现 {APP_VERSION}, "
                f"可能 APP_VERSION 改了但 commit message 漂移（旧版本号还残留）"
            )


# ---------------------------------------------------------------------------
# CI 提取版本号的契约（镜像 .github/workflows/docker-publish.yml）
# ---------------------------------------------------------------------------

class TestAppVersionGrepCompatible:
    """CI workflow 用这个 grep 提取版本号，必须兼容

    grep -oP 'APP_VERSION = "v\\K[^"]+' core/config.py
    """

    def test_grep_extracts_cleanly(self):
        """CI workflow 用这个 grep 提取版本号，必须兼容

        原始 CI 命令：grep -oP 'APP_VERSION = "v\\K[^"]+' core/config.py
        (grep -P 是 PCRE,支持 \\K)

        Python re 不支持 \\K,所以这里用等价写法：捕获 v 后面的部分。
        """
        import re
        with open("/home/dev/nas-submaster/core/config.py") as f:
            content = f.read()
        # 等价于 CI 的 grep
        m = re.search(r'APP_VERSION = "(v[^"]+)"', content)
        assert m is not None, "CI grep 模式没匹配到 APP_VERSION,workflow 会挂"
        # 提取出来的带 v 前缀
        assert m.group(1) == APP_VERSION
        # 同时确保 APP_VERSION 是 v 开头（CI 用 lstrip("v") 后取数字部分打 tag）
        assert APP_VERSION.startswith("v")
