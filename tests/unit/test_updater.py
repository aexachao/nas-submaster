#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
updater 模块单元测试
"""

import pytest
from unittest.mock import patch, MagicMock
from services.updater import (
    parse_version,
    compare_versions,
    get_latest_release,
    get_all_releases,
    has_update,
    do_update,
    ReleaseInfo,
    _build_docker_create_args,
)


# ============================================================================
# parse_version
# ============================================================================

class TestParseVersion:
    def test_standard_version(self):
        assert parse_version("v1.7.0") == (1, 7, 0)

    def test_without_v_prefix(self):
        assert parse_version("2.3.1") == (2, 3, 1)

    def test_single_digit(self):
        assert parse_version("v1") == (1,)

    def test_two_parts(self):
        assert parse_version("v1.7") == (1, 7)

    def test_invalid_version(self):
        assert parse_version("abc") == (0, 0, 0)

    def test_empty_string(self):
        assert parse_version("") == (0, 0, 0)

    def test_large_numbers(self):
        assert parse_version("v10.20.30") == (10, 20, 30)


# ============================================================================
# compare_versions
# ============================================================================

class TestCompareVersions:
    def test_equal(self):
        assert compare_versions("v1.6.0", "v1.6.0") == 0

    def test_current_older(self):
        assert compare_versions("v1.5.0", "v1.6.0") == -1

    def test_current_newer(self):
        assert compare_versions("v1.7.0", "v1.6.0") == 1

    def test_patch_difference(self):
        assert compare_versions("v1.6.0", "v1.6.1") == -1

    def test_major_difference(self):
        assert compare_versions("v1.0.0", "v2.0.0") == -1

    def test_with_and_without_v(self):
        assert compare_versions("v1.6.0", "1.6.0") == 0

    def test_invalid_versions(self):
        assert compare_versions("abc", "xyz") == 0


# ============================================================================
# ReleaseInfo
# ============================================================================

class TestReleaseInfo:
    def test_creation(self):
        info = ReleaseInfo(
            tag_name="v1.7.0",
            name="v1.7.0 - New Feature",
            body="## What's New\n- Feature A",
            published_at="2026-06-05T12:00:00Z",
            html_url="https://github.com/test/test/releases/tag/v1.7.0"
        )
        assert info.tag_name == "v1.7.0"
        assert "Feature A" in info.body


# ============================================================================
# get_latest_release (Docker Hub + GitHub fallback)
# ============================================================================

class TestGetLatestRelease:
    """
    2026-06-06 改造：get_latest_release() 优先用 Docker Registry v2 API
    （registry-1.docker.io），fallback 到 Docker Hub v2 API（hub.docker.com）。
    原因：Hub v2 有索引缓存延迟，CI 推完镜像后 5-30 分钟才可见。
    """

    def _mock_registry_v2_response(self, mock_get, tags: list):
        """helper: mock Registry v2 完整两步（auth + tags/list）"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake-token-abc123"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"name": "aexachao/nas-subtitle-manager", "tags": tags}
            else:
                # GitHub changelog
                resp.status_code = 200
                tag = url.split("/")[-1]
                resp.json.return_value = {
                    "tag_name": tag,
                    "name": f"{tag} - Release",
                    "body": f"Changes in {tag}",
                    "html_url": f"https://github.com/test/releases/tag/{tag}",
                }
            return resp
        mock_get.side_effect = side_effect

    @patch("services.updater.requests.get")
    def test_success_with_changelog(self, mock_get):
        """Registry v2 拿版本 + GitHub 拿更新日志，都成功"""
        self._mock_registry_v2_response(
            mock_get,
            tags=["v1.7.8", "v1.7.7", "v1.7.6", "latest"],
        )
        result = get_latest_release()
        assert result is not None
        assert result.tag_name == "v1.7.8"  # Registry v2 路径生效，取到最新
        assert "Release" in result.name
        assert "Changes in v1.7.8" in result.body

    @patch("services.updater.requests.get")
    def test_success_without_changelog(self, mock_get):
        """Registry v2 拿版本成功，GitHub 404（无更新日志）"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake-token"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"tags": ["v1.7.8", "latest"]}
            else:
                resp.status_code = 404
            return resp
        mock_get.side_effect = side_effect

        result = get_latest_release()
        assert result is not None
        assert result.tag_name == "v1.7.8"
        assert result.body == ""  # 无更新日志
        assert "v1.7.8" in result.html_url

    @patch("services.updater.requests.get")
    def test_registry_v2_fails_fallback_to_hub(self, mock_get):
        """Registry v2 失败时（认证失败），fallback 到 Docker Hub v2"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                # Registry auth 失败 → 模拟 Registry v2 不可用
                resp.status_code = 500
            elif "hub.docker.com" in url:
                # Fallback 到 Hub v2 成功
                resp.status_code = 200
                resp.json.return_value = {
                    "results": [
                        {"name": "v1.7.6"},
                        {"name": "v1.7.5"},
                        {"name": "latest"},
                    ]
                }
            else:
                resp.status_code = 200
                resp.json.return_value = {"tag_name": "v1.7.6", "name": "v1.7.6", "body": "", "html_url": ""}
            return resp
        mock_get.side_effect = side_effect

        result = get_latest_release()
        assert result is not None
        assert result.tag_name == "v1.7.6"  # 走了 fallback 拿到 Hub v2 的最新

    @patch("services.updater.requests.get")
    def test_both_apis_fail(self, mock_get):
        """Registry v2 + Hub v2 都失败 → 返回 None"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 500
            else:
                resp.status_code = 500
            return resp
        mock_get.side_effect = side_effect

        result = get_latest_release()
        assert result is None

    @patch("services.updater.requests.get")
    def test_no_version_tags(self, mock_get):
        """Registry v2 返回的 tag 列表里没有 v 前缀 tag"""
        self._mock_registry_v2_response(
            mock_get,
            tags=["latest", "abc123def"],  # 没有任何 v1.7.x
        )
        # Registry 返回空 version_tags → fallback 到 Hub v2 → Hub 也拿不到
        # 这里需要让 Hub v2 也失败
        def side_effect_2(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"tags": ["latest", "abc123def"]}
            else:
                # GitHub + Hub v2 都不行
                resp.status_code = 500
            return resp
        mock_get.side_effect = side_effect_2
        result = get_latest_release()
        assert result is None

    @patch("services.updater.requests.get")
    def test_network_error(self, mock_get):
        """完全网络错误 → 返回 None，不抛异常"""
        mock_get.side_effect = Exception("Network error")
        result = get_latest_release()
        assert result is None


# ============================================================================
# get_all_releases (Registry v2 主路径 + Hub v2 fallback)
# ============================================================================

class TestGetAllReleases:
    """
    2026-06-06 改造：get_all_releases() 优先用 Registry v2 拿全量 tag，
    fallback 到 Hub v2。
    """

    @patch("services.updater.requests.get")
    def test_success_from_registry_v2(self, mock_get):
        """Registry v2 路径：拉全量 tag → 按版本号降序 → 取前 N"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake-token"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                # 故意打乱顺序，测试排序逻辑
                resp.json.return_value = {
                    "tags": ["v1.7.6", "latest", "v1.7.8", "v1.7.7", "v1.6.0"]
                }
            else:
                tag = url.split("/")[-1]
                resp.status_code = 200
                resp.json.return_value = {
                    "tag_name": tag,
                    "name": f"{tag} - Release",
                    "body": f"Changes in {tag}",
                    "html_url": f"https://github.com/test/releases/tag/{tag}",
                }
            return resp
        mock_get.side_effect = side_effect

        result = get_all_releases(limit=3)
        assert len(result) == 3
        # 按版本号降序：v1.7.8 > v1.7.7 > v1.7.6
        assert result[0].tag_name == "v1.7.8"
        assert result[1].tag_name == "v1.7.7"
        assert result[2].tag_name == "v1.7.6"

    @patch("services.updater.requests.get")
    def test_fallback_to_hub_v2(self, mock_get):
        """Registry v2 失败 → fallback 到 Hub v2"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake-token"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                # Registry v2 返回空 tags 列表 → 触发 fallback
                resp.json.return_value = {"tags": ["latest"]}  # 无 v 前缀
            elif "hub.docker.com" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "results": [
                        {"name": "v1.7.6"},
                        {"name": "v1.7.5"},
                        {"name": "latest"},
                    ]
                }
            else:
                tag = url.split("/")[-1]
                resp.status_code = 200
                resp.json.return_value = {
                    "tag_name": tag, "name": f"{tag}", "body": "", "html_url": ""
                }
            return resp
        mock_get.side_effect = side_effect

        result = get_all_releases(limit=5)
        assert len(result) == 2  # v1.7.6 + v1.7.5（Hub v2 数据）
        assert result[0].tag_name == "v1.7.6"
        assert result[1].tag_name == "v1.7.5"

    @patch("services.updater.requests.get")
    def test_network_error(self, mock_get):
        """完全网络错误 → 返回空列表，不抛异常"""
        mock_get.side_effect = Exception("Network error")
        result = get_all_releases()
        assert result == []


# ============================================================================
# Registry v2 API 详细行为测试
# ============================================================================

class TestRegistryV2API:
    """
    2026-06-06 新增：直接测 _get_latest_version_from_registry() 和
    _fetch_all_version_tags_from_registry()，覆盖无索引缓存的"主路径"。
    """

    @patch("services.updater.requests.get")
    def test_auth_token_requested_with_correct_scope(self, mock_get):
        """调 Registry 前必须先拿 pull scope 的 token"""
        from services.updater import _get_latest_version_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "test-token"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"tags": ["v1.7.8", "v1.7.7"]}
            return resp
        mock_get.side_effect = side_effect
        _get_latest_version_from_registry()

        # 验证第一次调的是 auth.docker.io,且 params 含 scope=repository:...:pull
        first_call = mock_get.call_args_list[0]
        assert "auth.docker.io" in first_call.args[0]
        params = first_call.kwargs.get("params", {})
        assert params.get("scope") == "repository:aexachao/nas-subtitle-manager:pull"
        assert params.get("service") == "registry.docker.io"

    @patch("services.updater.requests.get")
    def test_registry_call_uses_bearer_token(self, mock_get):
        """调 registry-1.docker.io 时必须带 Bearer token header"""
        from services.updater import _get_latest_version_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "my-special-token-xyz"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"tags": ["v1.7.8"]}
            return resp
        mock_get.side_effect = side_effect
        _get_latest_version_from_registry()

        # 找 registry 这次调用
        registry_call = None
        for call in mock_get.call_args_list:
            if "registry-1.docker.io" in call.args[0]:
                registry_call = call
                break
        assert registry_call is not None
        headers = registry_call.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer my-special-token-xyz"

    @patch("services.updater.requests.get")
    def test_auth_fails_returns_none(self, mock_get):
        """auth.docker.io 失败 → Registry 路径返回 None（让上层 fallback）"""
        from services.updater import _get_latest_version_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 401  # 认证失败
            return resp
        mock_get.side_effect = side_effect
        result = _get_latest_version_from_registry()
        assert result is None

    @patch("services.updater.requests.get")
    def test_auth_returns_no_token(self, mock_get):
        """auth.docker.io 200 但响应里没 token → 返回 None"""
        from services.updater import _get_latest_version_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {}  # 没 token 字段
            return resp
        mock_get.side_effect = side_effect
        result = _get_latest_version_from_registry()
        assert result is None

    @patch("services.updater.requests.get")
    def test_registry_call_fails(self, mock_get):
        """registry-1.docker.io 失败 → 返回 None"""
        from services.updater import _get_latest_version_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 503
            return resp
        mock_get.side_effect = side_effect
        result = _get_latest_version_from_registry()
        assert result is None

    @patch("services.updater.requests.get")
    def test_filters_out_latest_and_non_version_tags(self, mock_get):
        """tag 列表里 latest 和非 v 前缀 tag 必须被过滤"""
        from services.updater import _get_latest_version_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "tags": [
                        "latest",            # 排除
                        "abc123def456",     # 排除（commit hash）
                        "v1.7.8",            # 保留
                        "v1.7.7",            # 保留
                        "v1.7.6",            # 保留
                        "v1.6.0-rc1",        # 保留
                    ]
                }
            return resp
        mock_get.side_effect = side_effect
        result = _get_latest_version_from_registry()
        assert result == "v1.7.8"  # max(parse_version) 选最高的

    @patch("services.updater.requests.get")
    def test_picks_max_version_not_first(self, mock_get):
        """必须按 parse_version 选最大,不能直接取 list[0]"""
        from services.updater import _get_latest_version_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                # 故意把 v1.7.8 放最前面，但 v1.10.0 才是最大
                resp.json.return_value = {
                    "tags": ["v1.7.8", "v1.10.0", "v1.9.5", "v1.7.6", "latest"]
                }
            return resp
        mock_get.side_effect = side_effect
        result = _get_latest_version_from_registry()
        assert result == "v1.10.0"  # 按数字比较，不是字符串

    @patch("services.updater.requests.get")
    def test_fetch_all_returns_full_list(self, mock_get):
        """_fetch_all_version_tags_from_registry() 返回完整列表,不是只 1 个"""
        from services.updater import _fetch_all_version_tags_from_registry

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "tags": ["v1.7.8", "v1.7.7", "v1.7.6", "v1.7.5", "latest", "abc123"]
                }
            return resp
        mock_get.side_effect = side_effect
        result = _fetch_all_version_tags_from_registry()
        # 不含 latest 和 commit hash
        assert set(result) == {"v1.7.8", "v1.7.7", "v1.7.6", "v1.7.5"}
        assert "latest" not in result

    @patch("services.updater.requests.get")
    def test_get_latest_version_prefers_registry(self, mock_get):
        """_get_latest_version() 必须先调 Registry v2 路径,成功就不调 Hub v2"""
        from services.updater import _get_latest_version

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"tags": ["v1.7.8"]}
            elif "hub.docker.com" in url:
                # 如果走到了 Hub v2,测试就失败（说明没走 Registry 主路径）
                raise AssertionError(
                    "不应该调用 hub.docker.com — Registry v2 应该先成功"
                )
            return resp
        mock_get.side_effect = side_effect

        result = _get_latest_version()
        assert result == "v1.7.8"

    @patch("services.updater.requests.get")
    def test_get_latest_version_fallback_when_registry_empty(self, mock_get):
        """Registry 拿到 version_tags=[] 时 → 触发 fallback"""
        from services.updater import _get_latest_version

        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "auth.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"token": "fake"}
            elif "registry-1.docker.io" in url:
                resp.status_code = 200
                resp.json.return_value = {"tags": ["latest", "abc123"]}  # 无 v 前缀
            elif "hub.docker.com" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "results": [{"name": "v1.7.6"}, {"name": "latest"}]
                }
            return resp
        mock_get.side_effect = side_effect
        result = _get_latest_version()
        assert result == "v1.7.6"  # 走了 fallback


# ============================================================================
# has_update
# ============================================================================

class TestHasUpdate:
    @patch("services.updater.get_latest_release")
    def test_has_update(self, mock_latest):
        mock_latest.return_value = ReleaseInfo(
            tag_name="v99.0.0",
            name="Future",
            body="",
            published_at="",
            html_url=""
        )
        assert has_update() is True

    @patch("services.updater.get_latest_release")
    def test_no_update(self, mock_latest):
        mock_latest.return_value = ReleaseInfo(
            tag_name="v0.0.1",
            name="Old",
            body="",
            published_at="",
            html_url=""
        )
        assert has_update() is False

    @patch("services.updater.get_latest_release")
    def test_api_failure(self, mock_latest):
        mock_latest.return_value = None
        assert has_update() is False


# ============================================================================
# do_update
# ============================================================================

MOCK_CONTAINER = {
    "Id": "abc123",
    "Name": "/test-container",
    "Config": {
        "Image": "aexachao/nas-subtitle-manager:latest",
        "Hostname": "test-host",
        "Env": ["TZ=Asia/Shanghai", "PYTHONUNBUFFERED=1"],
        "Labels": {"com.docker.compose.project": "nas"},
    },
    "HostConfig": {
        "Binds": ["./data:/data", "/var/run/docker.sock:/var/run/docker.sock"],
        "PortBindings": {"8501/tcp": [{"HostIp": "", "HostPort": "8501"}]},
        "RestartPolicy": {"Name": "unless-stopped", "MaximumRetryCount": 0},
        "NetworkMode": "bridge",
        "ShmSize": 4294967296,
    },
}


class TestDoUpdate:
    @patch("services.updater._docker_api")
    @patch("services.updater._get_container_info")
    @patch("services.updater.os.path.exists", return_value=True)
    def test_success(self, mock_exists, mock_info, mock_api):
        mock_info.return_value = MOCK_CONTAINER
        mock_api.side_effect = [
            (200, {}),                   # pull image
            (201, {"Id": "helper456"}),  # create helper
            (200, {}),                   # start helper
        ]
        ok, msg = do_update()
        assert ok is True
        assert "自动重启" in msg
        # 验证 helper 用的是 docker:cli 镜像
        create_call = mock_api.call_args_list[1]
        assert create_call[1]["body"]["Image"] == "docker:cli"
        assert create_call[1]["body"]["HostConfig"]["AutoRemove"] is True

    @patch("services.updater._docker_api")
    @patch("services.updater._get_container_info")
    @patch("services.updater.os.path.exists", return_value=True)
    def test_pull_failure(self, mock_exists, mock_info, mock_api):
        mock_info.return_value = MOCK_CONTAINER
        mock_api.return_value = (500, "pull failed")
        ok, msg = do_update()
        assert ok is False
        assert "拉取镜像失败" in msg

    @patch("services.updater._docker_api")
    @patch("services.updater._get_container_info")
    @patch("services.updater.os.path.exists", return_value=True)
    def test_helper_create_failure(self, mock_exists, mock_info, mock_api):
        mock_info.return_value = MOCK_CONTAINER
        mock_api.side_effect = [
            (200, {}),    # pull image
            (500, "err"), # create helper fails
        ]
        ok, msg = do_update()
        assert ok is False
        assert "更新助手" in msg

    @patch("services.updater.os.path.exists", return_value=False)
    def test_socket_not_found(self, mock_exists):
        ok, msg = do_update()
        assert ok is False
        assert "Socket" in msg

    @patch("services.updater._get_container_info")
    @patch("services.updater.os.path.exists", return_value=True)
    def test_container_info_failure(self, mock_exists, mock_info):
        mock_info.return_value = None
        ok, msg = do_update()
        assert ok is False
        assert "容器信息" in msg


class TestBuildDockerCreateArgs:
    def test_basic_config(self):
        args = _build_docker_create_args(MOCK_CONTAINER, "myimage:latest")
        assert "--name" in args
        assert "test-container" in args
        assert "myimage:latest" in args[-1]

    def test_preserves_env(self):
        args = _build_docker_create_args(MOCK_CONTAINER, "img")
        assert "-e" in args
        assert "TZ=Asia/Shanghai" in args

    def test_preserves_volumes(self):
        args = _build_docker_create_args(MOCK_CONTAINER, "img")
        assert "-v" in args
        assert "./data:/data" in args

    def test_preserves_ports(self):
        args = _build_docker_create_args(MOCK_CONTAINER, "img")
        assert "-p" in args
        assert "8501:8501" in args

    def test_preserves_restart_policy(self):
        args = _build_docker_create_args(MOCK_CONTAINER, "img")
        assert "--restart" in args
        assert "unless-stopped" in args

    def test_preserves_shm_size(self):
        args = _build_docker_create_args(MOCK_CONTAINER, "img")
        assert "--shm-size" in args

    def test_empty_config(self):
        container = {
            "Name": "/minimal",
            "Config": {"Image": "img"},
            "HostConfig": {},
        }
        args = _build_docker_create_args(container, "img")
        assert "--name" in args
        assert "minimal" in args
        assert args[-1] == "img"
