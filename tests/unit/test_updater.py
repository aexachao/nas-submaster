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
    @patch("services.updater.requests.get")
    def test_success_with_changelog(self, mock_get):
        """Docker Hub 版本 + GitHub 更新日志都成功"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "hub.docker.com" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "results": [
                        {"name": "v1.7.1"},
                        {"name": "v1.7.0"},
                        {"name": "latest"},
                    ]
                }
            else:
                resp.status_code = 200
                resp.json.return_value = {
                    "tag_name": "v1.7.1",
                    "name": "v1.7.1 - Bug Fix",
                    "body": "修复超时问题",
                    "html_url": "https://github.com/test/releases/tag/v1.7.1"
                }
            return resp

        mock_get.side_effect = side_effect
        result = get_latest_release()
        assert result is not None
        assert result.tag_name == "v1.7.1"
        assert "Bug Fix" in result.name
        assert "超时" in result.body

    @patch("services.updater.requests.get")
    def test_success_without_changelog(self, mock_get):
        """Docker Hub 成功，GitHub 失败（仅版本号，无更新日志）"""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "hub.docker.com" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "results": [
                        {"name": "v1.7.1"},
                        {"name": "latest"},
                    ]
                }
            else:
                resp.status_code = 404
            return resp

        mock_get.side_effect = side_effect
        result = get_latest_release()
        assert result is not None
        assert result.tag_name == "v1.7.1"
        assert result.body == ""  # 无更新日志

    @patch("services.updater.requests.get")
    def test_docker_hub_error(self, mock_get):
        """Docker Hub 失败，返回 None"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        result = get_latest_release()
        assert result is None

    @patch("services.updater.requests.get")
    def test_no_version_tags(self, mock_get):
        """Docker Hub 没有版本标签"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [{"name": "latest"}]
        }
        mock_get.return_value = mock_resp

        result = get_latest_release()
        assert result is None

    @patch("services.updater.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_latest_release()
        assert result is None


# ============================================================================
# get_all_releases (Docker Hub + GitHub fallback)
# ============================================================================

class TestGetAllReleases:
    @patch("services.updater.requests.get")
    def test_success(self, mock_get):
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "hub.docker.com" in url:
                resp.status_code = 200
                resp.json.return_value = {
                    "results": [
                        {"name": "v1.7.1"},
                        {"name": "v1.7.0"},
                        {"name": "v1.6.0"},
                        {"name": "latest"},
                    ]
                }
            else:
                # GitHub changelog for each tag
                tag = url.split("/")[-1]
                resp.status_code = 200
                resp.json.return_value = {
                    "tag_name": tag,
                    "name": f"{tag} - Release",
                    "body": f"Changes in {tag}",
                    "html_url": f"https://github.com/test/releases/tag/{tag}"
                }
            return resp

        mock_get.side_effect = side_effect
        result = get_all_releases(limit=3)
        assert len(result) == 3
        assert result[0].tag_name == "v1.7.1"
        assert result[1].tag_name == "v1.7.0"
        assert result[2].tag_name == "v1.6.0"

    @patch("services.updater.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_all_releases()
        assert result == []


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
