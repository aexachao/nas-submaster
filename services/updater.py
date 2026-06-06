#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用更新服务
从 Docker Hub 检查新版本、从 GitHub 获取更新日志、执行 Docker 更新
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import requests

from core.config import APP_VERSION

# Docker Hub 仓库信息
DOCKER_HUB_REPO = "aexachao/nas-subtitle-manager"
DOCKER_HUB_API = f"https://hub.docker.com/v2/repositories/{DOCKER_HUB_REPO}/tags"

# GitHub 仓库信息（仅用于获取更新日志）
GITHUB_OWNER = "aexachao"
GITHUB_REPO = "nas-submaster"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"

# 版本号标签前缀
VERSION_TAG_PREFIX = "v"


@dataclass
class ReleaseInfo:
    """版本信息"""
    tag_name: str       # 如 "v1.7.0"
    name: str           # 如 "v1.7.0 - 新增自动更新"
    body: str           # 更新日志正文（可能为空）
    published_at: str   # 发布时间
    html_url: str       # GitHub Release 链接


def parse_version(version_str: str) -> tuple:
    """
    解析版本号字符串为可比较的元组。

    Args:
        version_str: 如 "v1.7.0" 或 "1.7.0"
    Returns:
        (major, minor, patch) 如 (1, 7, 0)
    """
    v = version_str.lstrip("v")
    parts = v.split(".")
    try:
        return tuple(int(p) for p in parts[:3])
    except (ValueError, IndexError):
        return (0, 0, 0)


def compare_versions(current: str, latest: str) -> int:
    """
    比较两个版本号。

    Returns:
        -1: current < latest（有更新）
         0: current == latest（已是最新）
         1: current > latest（当前版本更新，测试场景）
    """
    c = parse_version(current)
    l = parse_version(latest)
    if c < l:
        return -1
    elif c > l:
        return 1
    return 0


def _get_latest_version_from_dockerhub() -> Optional[str]:
    """
    从 Docker Hub 获取最新版本号标签。

    Returns:
        版本号字符串（如 "v1.7.1"）或 None
    """
    try:
        resp = requests.get(
            DOCKER_HUB_API,
            params={"page_size": 50, "ordering": "-last_updated"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None

        tags = resp.json().get("results", [])
        version_tags = [
            t["name"] for t in tags
            if t["name"].startswith(VERSION_TAG_PREFIX)
            and t["name"] != "latest"
        ]

        if not version_tags:
            return None

        # 找到最新版本
        return max(version_tags, key=lambda t: parse_version(t))

    except Exception as e:
        print(f"[Updater] Failed to check Docker Hub: {e}")
        return None


def _get_changelog_from_github(tag_name: str) -> tuple:
    """
    从 GitHub 获取指定版本的更新日志（可选，失败不影响主流程）。

    Args:
        tag_name: 版本号标签（如 "v1.7.1"）
    Returns:
        (name, body, html_url) 或 ("", "", "")
    """
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/releases/tags/{tag_name}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return (
                data.get("name", ""),
                data.get("body", ""),
                data.get("html_url", ""),
            )
    except Exception:
        pass
    return ("", "", "")


def get_latest_release() -> Optional[ReleaseInfo]:
    """
    获取最新版本信息。
    版本号从 Docker Hub 获取，更新日志从 GitHub 获取（可选）。

    Returns:
        ReleaseInfo 或 None（网络错误时）
    """
    tag_name = _get_latest_version_from_dockerhub()
    if not tag_name:
        return None

    # 尝试从 GitHub 补充更新日志（失败不影响主流程）
    name, body, html_url = _get_changelog_from_github(tag_name)
    if not name:
        name = tag_name
    if not html_url:
        html_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tag/{tag_name}"

    return ReleaseInfo(
        tag_name=tag_name,
        name=name,
        body=body,
        published_at="",
        html_url=html_url,
    )


def get_all_releases(limit: int = 10) -> List[ReleaseInfo]:
    """
    获取最近 N 个版本的信息。

    Args:
        limit: 最多返回条数
    Returns:
        ReleaseInfo 列表
    """
    try:
        resp = requests.get(
            DOCKER_HUB_API,
            params={"page_size": 50, "ordering": "-last_updated"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []

        tags = resp.json().get("results", [])
        version_tags = [
            t["name"] for t in tags
            if t["name"].startswith(VERSION_TAG_PREFIX)
            and t["name"] != "latest"
        ]

        # 按版本号降序排列
        version_tags.sort(key=lambda t: parse_version(t), reverse=True)
        version_tags = version_tags[:limit]

        results = []
        for tag in version_tags:
            name, body, html_url = _get_changelog_from_github(tag)
            if not name:
                name = tag
            if not html_url:
                html_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tag/{tag}"
            results.append(ReleaseInfo(
                tag_name=tag,
                name=name,
                body=body,
                published_at="",
                html_url=html_url,
            ))

        return results

    except Exception as e:
        print(f"[Updater] Failed to fetch releases: {e}")
        return []


def has_update() -> bool:
    """检查是否有新版本可用"""
    latest = get_latest_release()
    if latest is None:
        return False
    return compare_versions(APP_VERSION, latest.tag_name) < 0


DOCKER_SOCKET = "/var/run/docker.sock"


def _docker_api(method: str, path: str, body: dict = None, timeout: int = 300) -> tuple:
    """
    通过 Docker Socket 调用 Docker Engine API。

    Args:
        method: HTTP 方法（GET/POST）
        path: API 路径（如 /images/json）
        body: 请求体（POST 时使用）
        timeout: 超时秒数
    Returns:
        (status_code, response_json_or_text)
    """
    import http.client
    import json

    conn = http.client.HTTPUnixConnection(DOCKER_SOCKET)
    try:
        headers = {"Content-Type": "application/json"}
        payload = json.dumps(body) if body else None
        conn.request(method, f"http://localhost{path}", body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", errors="replace")
        try:
            return resp.status, json.loads(data)
        except json.JSONDecodeError:
            return resp.status, data
    finally:
        conn.close()


def _get_container_info() -> Optional[dict]:
    """获取当前容器信息（通过 hostname 即容器 ID）"""
    import socket
    hostname = socket.gethostname()
    status, data = _docker_api("GET", f"/containers/{hostname}/json")
    if status == 200:
        return data
    return None


def do_update() -> tuple:
    """
    通过 Docker Engine API 拉取最新镜像。
    容器重启由用户在宿主机执行 docker compose up -d 完成，
    以保证用户自己的 docker-compose.yml 配置和数据不受影响。

    Returns:
        (success: bool, message: str)
    """
    try:
        # 检查 Docker Socket 是否可用
        if not os.path.exists(DOCKER_SOCKET):
            return False, (
                "未找到 Docker Socket，请在 docker-compose.yml 中挂载：\n"
                "volumes:\n"
                "  - /var/run/docker.sock:/var/run/docker.sock"
            )

        # 获取当前容器信息
        container = _get_container_info()
        if not container:
            return False, "无法获取当前容器信息"

        # 提取镜像名
        image = container.get("Config", {}).get("Image", "")
        if not image:
            return False, "无法获取当前容器镜像名"

        # 只拉取最新镜像，不触碰正在运行的容器
        status, resp = _docker_api("POST", f"/images/create?fromImage={image}&tag=latest")
        if status not in (200, 201):
            return False, f"拉取镜像失败: {resp}"

        return True, (
            "镜像已更新！请在宿主机执行以下命令完成重启：\n"
            "docker compose pull && docker compose up -d"
        )

    except ConnectionRefusedError:
        return False, "无法连接 Docker Socket，请确认已挂载 /var/run/docker.sock"
    except Exception as e:
        return False, f"更新失败: {e}"
