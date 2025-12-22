#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字幕翻译模块（重构版）
主要改进：
1. 使用 JSON 格式强制结构化输出
2. 智能分段策略（短视频整体翻译，长视频分场景）
3. 完善错误处理（不再静默失败）
4. 支持翻译质量检测和重试
"""

import json
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class TranslationConfig:
    """翻译配置"""
    api_key: str
    base_url: str
    model_name: str
    target_language: str
    source_language: str = 'auto'
    max_lines_per_batch: int = 500  # 每批最多翻译多少行
    max_retries: int = 3
    timeout: int = 180


class SubtitleEntry:
    """字幕条目"""
    def __init__(self, index: str, timecode: str, text: str):
        self.index = index
        self.timecode = timecode
        self.text = text
    
    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'timecode': self.timecode,
            'text': self.text
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SubtitleEntry':
        return cls(
            index=str(data.get('index', '1')),
            timecode=str(data.get('timecode', '00:00:00,000 --> 00:00:01,000')),
            text=str(data.get('text', ''))
        )


class TranslationError(Exception):
    """翻译错误基类"""
    pass


class APIError(TranslationError):
    """API 调用错误"""
    pass


class ParseError(TranslationError):
    """解析错误"""
    pass


class SubtitleTranslator:
    """字幕翻译器"""
    
    # 语言名称映射
    LANG_NAMES = {
        'zh': '中文 (Simplified Chinese)',
        'en': '英语 (English)',
        'ja': '日语 (Japanese)',
        'ko': '韩语 (Korean)',
        'fr': '法语 (French)',
        'de': '德语 (German)',
        'ru': '俄语 (Russian)',
        'es': '西班牙语 (Spanish)'
    }
    
    def __init__(self, config: TranslationConfig, progress_callback=None):
        """
        初始化翻译器
        
        Args:
            config: 翻译配置
            progress_callback: 进度回调函数 (current, total, message)
        """
        self.config = config
        self.progress_callback = progress_callback
        
        # 初始化 OpenAI 客户端
        api_key = config.api_key
        if "ollama" in config.base_url.lower():
            api_key = "ollama"
        
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)
    
    def _update_progress(self, current: int, total: int, message: str):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(current, total, message)
    
    def _get_target_lang_name(self) -> str:
        """获取目标语言名称"""
        return self.LANG_NAMES.get(
            self.config.target_language, 
            self.config.target_language
        )
    
    def _build_translation_prompt(
        self, 
        entries: List[SubtitleEntry],
        context_before: Optional[str] = None,
        context_after: Optional[str] = None
    ) -> str:
        """
        构建翻译 prompt（使用 JSON 格式）
        
        Args:
            entries: 要翻译的字幕条目
            context_before: 前文上下文（仅供参考，不翻译）
            context_after: 后文上下文（仅供参考，不翻译）
        """
        target_lang = self._get_target_lang_name()
        
        # 构建输入 JSON
        input_json = [
            {"line": i+1, "text": entry.text}
            for i, entry in enumerate(entries)
        ]
        
        # 上下文提示
        context_hint = ""
        if context_before or context_after:
            context_hint = "\n\nCONTEXT (for reference only, helps understand the flow):"
            if context_before:
                context_hint += f"\nPrevious line: \"{context_before}\""
            if context_after:
                context_hint += f"\nNext line: \"{context_after}\""
        
        prompt = f"""You are a professional subtitle translator. Translate the following dialogue to {target_lang}.

CRITICAL RULES:
1. Output MUST be valid JSON array with EXACTLY {len(entries)} objects
2. Each object MUST have "line" (number) and "translation" (string) fields
3. Keep translations natural and concise - match the style of {target_lang} subtitles
4. Preserve character names, proper nouns, and technical terms appropriately
5. DO NOT add punctuation at the end unless present in the original
6. DO NOT merge, split, or skip any lines
7. If a line is untranslatable (music notes, sound effects), keep it as-is
8. IMPORTANT: Output COMPLETE JSON array - DO NOT use "..." to abbreviate{context_hint}

INPUT ({len(entries)} lines):
{json.dumps(input_json, ensure_ascii=False, indent=2)}

OUTPUT FORMAT (valid JSON array with ALL {len(entries)} items):
[
  {{"line": 1, "translation": "..."}},
  {{"line": 2, "translation": "..."}},
  {{"line": 3, "translation": "..."}},
  {{"line": {len(entries)}, "translation": "..."}}
]

REMINDER: You MUST include ALL {len(entries)} translations. DO NOT abbreviate with "..." in the actual output.

Now output the COMPLETE JSON array (no extra text, no abbreviations):"""
        
        return prompt
    
    def _parse_translation_response(
        self, 
        response: str, 
        expected_count: int
    ) -> List[str]:
        """
        解析翻译响应（JSON 格式）- 增强版
        
        Args:
            response: API 返回的原始响应
            expected_count: 期望的翻译数量
        
        Returns:
            翻译文本列表
        
        Raises:
            ParseError: 解析失败
        """
        # 清理响应（移除可能的 markdown 代码块）
        response = response.strip()
        if response.startswith("```"):
            # 移除 ```json 或 ``` 开头
            lines = response.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        # 移除可能的前缀文本（如 "好的，以下是翻译："）
        if not response.strip().startswith('['):
            # 找到第一个 [ 的位置
            bracket_pos = response.find('[')
            if bracket_pos > 0:
                response = response[bracket_pos:]
        
        # 检查是否包含省略符号 "..."
        if '...' in response and '"...' not in response:
            # 如果 ... 出现在非字符串中（即作为省略符号），说明 AI 返回了缩略格式
            raise ParseError(
                f"AI 返回了省略格式（包含 ...），请求的 {expected_count} 条翻译未完整返回。"
                "这通常是因为内容过长，请降低 max_lines_per_batch 配置。"
            )
        
        # 尝试解析 JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            # JSON 解析失败，尝试修复常见问题
            
            # 尝试 1: 移除尾部逗号（如果有）
            if response.rstrip().endswith(',]'):
                response = response.rstrip()[:-2] + ']'
                try:
                    data = json.loads(response)
                except:
                    pass
            
            # 尝试 2: 检查是否缺少闭合括号
            if not response.rstrip().endswith(']'):
                response = response.rstrip() + ']'
                try:
                    data = json.loads(response)
                except:
                    pass
            
            # 如果仍然失败，抛出详细错误
            if 'data' not in locals():
                raise ParseError(
                    f"JSON 解析失败: {e}\n"
                    f"原始响应预览: {response[:300]}...\n"
                    f"响应长度: {len(response)} 字符"
                )
        
        # 验证格式
        if not isinstance(data, list):
            raise ParseError(f"期望 JSON 数组，实际得到: {type(data).__name__}")
        
        if len(data) != expected_count:
            raise ParseError(
                f"翻译数量不匹配: 期望 {expected_count} 条，实际 {len(data)} 条\n"
                f"提示: 如果 AI 返回了省略格式，请降低 max_lines_per_batch 配置（当前可能过大）"
            )
        
        # 提取翻译文本
        translations = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ParseError(f"第 {i+1} 项不是字典: {item}")
            
            if 'translation' not in item:
                raise ParseError(f"第 {i+1} 项缺少 'translation' 字段: {item}")
            
            line_num = item.get('line', i+1)
            if line_num != i+1:
                raise ParseError(
                    f"第 {i+1} 项的 line 值错误: 期望 {i+1}，实际 {line_num}"
                )
            
            translations.append(str(item['translation']).strip())
        
        return translations
    
    def _translate_batch(
        self,
        entries: List[SubtitleEntry],
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
        retry_count: int = 0
    ) -> List[SubtitleEntry]:
        """
        翻译一批字幕（带重试和智能降级）
        
        Args:
            entries: 要翻译的字幕条目
            context_before: 前文上下文
            context_after: 后文上下文
            retry_count: 当前重试次数（用于智能降级）
        
        Returns:
            翻译后的字幕条目
        
        Raises:
            APIError: API 调用失败
            ParseError: 解析失败
        """
        # 智能降级：如果批次过大导致 AI 返回省略格式，自动拆分
        if len(entries) > 100 and retry_count >= 2:
            print(f"[智能降级] 批次过大 ({len(entries)} 行)，自动拆分为 2 个子批次")
            mid = len(entries) // 2
            batch1 = self._translate_batch(entries[:mid], context_before, entries[mid].text, 0)
            batch2 = self._translate_batch(entries[mid:], entries[mid-1].text, context_after, 0)
            return batch1 + batch2
        
        prompt = self._build_translation_prompt(entries, context_before, context_after)
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=self.config.timeout
                )
                
                raw_response = response.choices[0].message.content.strip()
                translations = self._parse_translation_response(raw_response, len(entries))
                
                # 构建翻译后的字幕条目
                translated_entries = []
                for entry, translation in zip(entries, translations):
                    translated_entries.append(
                        SubtitleEntry(
                            index=entry.index,
                            timecode=entry.timecode,
                            text=translation
                        )
                    )
                
                return translated_entries
                
            except ParseError as e:
                last_error = e
                error_msg = str(e)
                
                # 检查是否是省略格式导致的错误
                if "省略格式" in error_msg or "..." in error_msg:
                    print(f"[翻译] 检测到省略格式，批次大小: {len(entries)} 行")
                    # 触发智能降级
                    if len(entries) > 50:
                        return self._translate_batch(entries, context_before, context_after, retry_count + 99)
                
                if attempt < self.config.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[翻译] 解析失败，{wait_time}秒后重试 ({attempt+1}/{self.config.max_retries}): {error_msg[:100]}")
                    time.sleep(wait_time)
                else:
                    raise
            
            except Exception as e:
                last_error = APIError(f"API 调用失败: {e}")
                if attempt < self.config.max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[翻译] API 错误，{wait_time}秒后重试 ({attempt+1}/{self.config.max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    raise last_error
        
        # 不应该到达这里
        raise last_error or TranslationError("翻译失败，原因未知")
    
    def translate_subtitles(
        self, 
        entries: List[SubtitleEntry]
    ) -> List[SubtitleEntry]:
        """
        翻译字幕（智能分段）
        
        Args:
            entries: 原始字幕条目列表
        
        Returns:
            翻译后的字幕条目列表
        """
        if not entries:
            return []
        
        total_lines = len(entries)
        max_batch = self.config.max_lines_per_batch
        
        # 短视频：一次性翻译
        if total_lines <= max_batch:
            self._update_progress(0, total_lines, f"开始翻译 {total_lines} 行字幕...")
            
            try:
                translated = self._translate_batch(entries)
                self._update_progress(total_lines, total_lines, "翻译完成！")
                return translated
            except Exception as e:
                raise TranslationError(f"翻译失败: {e}")
        
        # 长视频：分批翻译（保留上下文）
        translated_entries = []
        total_batches = (total_lines + max_batch - 1) // max_batch
        
        for i in range(0, total_lines, max_batch):
            batch_num = i // max_batch + 1
            batch = entries[i:i+max_batch]
            
            # 获取上下文
            context_before = entries[i-1].text if i > 0 else None
            context_after = entries[i+max_batch].text if i+max_batch < total_lines else None
            
            self._update_progress(
                i, 
                total_lines, 
                f"正在翻译第 {batch_num}/{total_batches} 批（{len(batch)} 行）..."
            )
            
            try:
                translated_batch = self._translate_batch(batch, context_before, context_after)
                translated_entries.extend(translated_batch)
            except Exception as e:
                raise TranslationError(
                    f"第 {batch_num}/{total_batches} 批翻译失败: {e}"
                )
        
        self._update_progress(total_lines, total_lines, "翻译完成！")
        return translated_entries


# ============================================================================
# 辅助函数
# ============================================================================

def parse_srt_file(srt_path: str) -> List[SubtitleEntry]:
    """
    解析 SRT 文件
    
    Args:
        srt_path: SRT 文件路径
    
    Returns:
        字幕条目列表
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entries = []
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                entries.append(
                    SubtitleEntry(
                        index=lines[0].strip(),
                        timecode=lines[1].strip(),
                        text='\n'.join(lines[2:]).strip()
                    )
                )
            except Exception as e:
                print(f"[警告] 跳过无效字幕块: {e}")
                continue
    
    return entries


def save_srt_file(entries: List[SubtitleEntry], output_path: str):
    """
    保存 SRT 文件
    
    Args:
        entries: 字幕条目列表
        output_path: 输出文件路径
    """
    lines = []
    for entry in entries:
        if not entry.text:
            continue
        lines.append(f"{entry.index}\n{entry.timecode}\n{entry.text}\n")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def translate_srt_file(
    input_path: str,
    config: TranslationConfig,
    output_path: Optional[str] = None,
    progress_callback=None
) -> Tuple[bool, str]:
    """
    翻译 SRT 文件（高级封装）
    
    Args:
        input_path: 输入 SRT 文件路径
        config: 翻译配置
        output_path: 输出路径（默认：原文件名.{target_lang}.srt）
        progress_callback: 进度回调
    
    Returns:
        (成功标志, 消息)
    """
    try:
        # 解析原始字幕
        entries = parse_srt_file(input_path)
        if not entries:
            return False, "字幕文件为空或格式错误"
        
        # 执行翻译
        translator = SubtitleTranslator(config, progress_callback)
        translated_entries = translator.translate_subtitles(entries)
        
        # 生成输出路径
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(
                input_file.parent / 
                f"{input_file.stem}.{config.target_language}.srt"
            )
        
        # 保存翻译结果
        save_srt_file(translated_entries, output_path)
        
        return True, f"翻译完成，已保存到: {output_path}"
        
    except TranslationError as e:
        return False, f"翻译失败: {e}"
    except Exception as e:
        return False, f"未知错误: {e}"