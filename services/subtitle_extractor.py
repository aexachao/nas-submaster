#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内置字幕提取服务
使用 ffprobe 检测字幕轨道，ffmpeg 提取字幕
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from core.models import SubtitleTrack


class SubtitleExtractor:
    """内置字幕提取器"""

    @staticmethod
    def detect_subtitle_tracks(video_path: str) -> List[SubtitleTrack]:
        """
        检测视频中的字幕轨道

        使用 ffprobe 获取字幕流信息：
        ffprobe -v quiet -print_format json -show_streams -select_streams s input.mkv

        Args:
            video_path: 视频文件路径

        Returns:
            字幕轨道列表
        """
        tracks = []

        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_streams',
                '-select_streams', 's',
                video_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return []

            data = json.loads(result.stdout)
            streams = data.get('streams', [])

            for idx, stream in enumerate(streams):
                # 获取语言标签（可能在大写或小写 keys 中）
                language = ''
                for key in ['tags', 'codec_tag_string', 'codec_name']:
                    if key == 'tags':
                        tags = stream.get('tags', {})
                        language = tags.get('language', '') or tags.get('LANGUAGE', '')
                        if not language:
                            # 尝试从 disposition 获取默认轨道信息
                            disposition = stream.get('disposition', {})
                            if disposition.get('default'):
                                language = 'default'
                    elif key == 'codec_tag_string':
                        codec_name = stream.get('codec_tag_string', '')
                    elif key == 'codec_name':
                        codec_name = stream.get('codec_name', '')

                if not language:
                    language = 'unknown'

                # 获取轨道标题
                tags = stream.get('tags', {})
                title = tags.get('title', '')

                # 判断是否为软字幕（ASS/SSA/SRT/VTT 等格式通常是软字幕）
                codec_name = stream.get('codec_name', '')
                is_soft = codec_name.lower() in ['srt', 'ass', 'ssa', 'subrip', 'vtt', 'sub']

                track = SubtitleTrack(
                    stream_index=idx,
                    codec_name=codec_name,
                    language=language,
                    title=title,
                    is_soft_subtitle=is_soft
                )
                tracks.append(track)

        except subprocess.TimeoutExpired:
            print(f"[SubtitleExtractor] ffprobe timeout for {video_path}")
        except json.JSONDecodeError:
            print(f"[SubtitleExtractor] Failed to parse ffprobe output for {video_path}")
        except Exception as e:
            print(f"[SubtitleExtractor] Error detecting subtitles: {e}")

        return tracks

    @staticmethod
    def extract_subtitle(
        video_path: str,
        track_index: int,
        output_path: Optional[str] = None,
        output_format: str = 'srt',
        embedded: bool = False
    ) -> Optional[str]:
        """
        提取指定轨道字幕

        使用 ffmpeg 按字幕流提取：
        ffmpeg -i input.mkv -map 0:s:0 -c:s srt output.srt

        Args:
            video_path: 视频文件路径
            track_index: 字幕轨道索引（从 0 开始）
            output_path: 输出文件路径，默认在视频同目录下生成
            output_format: 输出格式 (srt, ass, etc.)
            embedded: 是否为内置字幕，True 时文件名加上 .embedded 标记

        Returns:
            提取后的文件路径，失败返回 None
        """
        if output_path is None:
            video_dir = Path(video_path).parent
            video_stem = Path(video_path).stem
            # 内置字幕加上 embedded 标记，便于后续识别
            if embedded:
                output_path = str(video_dir / f"{video_stem}.embedded.{output_format}")
            else:
                output_path = str(video_dir / f"{video_stem}.{output_format}")

        try:
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-i', video_path,
                '-map', f'0:s:{track_index}',
                '-c:s', output_format,
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0 and Path(output_path).exists():
                return output_path
            else:
                print(f"[SubtitleExtractor] ffmpeg failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"[SubtitleExtractor] ffmpeg timeout for {video_path}")
            return None
        except Exception as e:
            print(f"[SubtitleExtractor] Error extracting subtitle: {e}")
            return None

    @staticmethod
    def has_embedded_subtitles(video_path: str) -> bool:
        """
        检查视频是否包含可提取的内置字幕

        Args:
            video_path: 视频文件路径

        Returns:
            True 如果有软字幕轨道
        """
        tracks = SubtitleExtractor.detect_subtitle_tracks(video_path)
        return any(t.is_soft_subtitle for t in tracks)

    @staticmethod
    def select_best_subtitle_track(
        tracks: List[SubtitleTrack],
        target_language: str = 'zh'
    ) -> Optional[SubtitleTrack]:
        """
        选择最佳字幕轨道

        优先选择非目标语言的第一条字幕轨道

        Args:
            tracks: 字幕轨道列表
            target_language: 目标语言代码

        Returns:
            最佳字幕轨道，如果没有合适的选择返回 None
        """
        if not tracks:
            return None

        # 过滤软字幕
        soft_tracks = [t for t in tracks if t.is_soft_subtitle]
        if not soft_tracks:
            return None

        # 优先选择非目标语言的字幕
        non_target = [t for t in soft_tracks if t.language != target_language]
        if non_target:
            # 返回第一条非目标语言的字幕
            return non_target[0]

        # 如果没有非目标语言字幕，返回第一条软字幕
        return soft_tracks[0]


def detect_subtitle_tracks(video_path: str) -> List[SubtitleTrack]:
    """快捷函数：检测字幕轨道"""
    return SubtitleExtractor.detect_subtitle_tracks(video_path)


def extract_subtitle(
    video_path: str,
    track_index: int,
    output_path: Optional[str] = None,
    output_format: str = 'srt'
) -> Optional[str]:
    """快捷函数：提取字幕"""
    return SubtitleExtractor.extract_subtitle(video_path, track_index, output_path, output_format)


def has_embedded_subtitles(video_path: str) -> bool:
    """快捷函数：检查是否有内置字幕"""
    return SubtitleExtractor.has_embedded_subtitles(video_path)
