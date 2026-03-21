"""
MiMo-V2-TTS 流式语音合成示例
通过 chat.completions 接口 + audio 参数实现 TTS 流式生成
"""

import os
import sys
import io
import json
import struct
import base64
import argparse
from pathlib import Path
from openai import OpenAI


def _extract_pcm_from_wav(wav_bytes: bytes) -> bytes:
    """
    从完整的 WAV 文件中提取纯 PCM 数据。
    每个 chunk 返回的 audio.data 都是一个完整 WAV（带 RIFF 头），
    直接拼接会导致只有第一段播放。必须剥掉头部只取 PCM 数据。
    """
    # 查找 "data" 子块的位置
    idx = wav_bytes.find(b"data")
    if idx == -1:
        # 没有 data 标记，可能不是标准 WAV，原样返回
        return wav_bytes
    # "data" 后 4 字节是 data chunk 的大小，再之后是实际 PCM 数据
    data_start = idx + 8  # skip "data" (4) + size (4)
    return wav_bytes[data_start:]


def _build_wav_header(pcm_size: int, sample_rate: int = 24000,
                      channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """构造标准 WAV 文件头"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + pcm_size,       # 文件总大小 - 8
        b"WAVE",
        b"fmt ",
        16,                  # fmt chunk 大小
        1,                   # PCM 格式
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        pcm_size,
    )
    return header


def _parse_wav_params(wav_bytes: bytes) -> tuple[int, int, int]:
    """从 WAV 头中读取 sample_rate, channels, bits_per_sample"""
    try:
        # fmt chunk 从偏移 12 开始: "fmt " (4) + size (4) + format (2) + channels (2) + sample_rate (4) ...
        channels = struct.unpack_from("<H", wav_bytes, 22)[0]
        sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
        bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
        return sample_rate, channels, bits_per_sample
    except Exception:
        return 24000, 1, 16


def stream_tts(
    text: str,
    output_path: str = "output.wav",
    voice: str = "mimo_default",
    audio_format: str = "wav",
    api_key: str | None = None,
    base_url: str = "https://api.xiaomimimo.com/v1",
    debug: bool = False,
):
    """
    调用 MiMo-V2-TTS API，流式生成音频。
    """
    api_key = api_key or os.getenv("MIMO_API_KEY")
    if not api_key:
        print("错误: 请设置 MIMO_API_KEY 环境变量，或通过 --api-key 参数传入。")
        print("  获取地址: https://platform.xiaomimimo.com/#/console/api-keys")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"正在合成: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
    print(f"语音风格: {voice}  |  格式: {audio_format}  |  输出: {output_path}")

    messages = [
        {"role": "user", "content": "请朗读以下内容"},
        {"role": "assistant", "content": text},
    ]

    print("开始流式接收...")

    completion = client.chat.completions.create(
        model="mimo-v2-tts",
        messages=messages,
        audio={
            "format": audio_format,
            "voice": voice,
        },
        stream=True,
    )

    # 收集 PCM 数据（每个 chunk 的 WAV 头会被剥掉）
    pcm_parts = []
    audio_chunk_count = 0
    chunk_count = 0
    sample_rate, channels, bits_per_sample = 24000, 1, 16  # 默认参数

    for chunk in completion:
        chunk_count += 1

        if debug and chunk_count <= 5:
            chunk_json = chunk.model_dump_json()
            preview = chunk_json[:300] + "..." if len(chunk_json) > 300 else chunk_json
            print(f"\n  [DEBUG] chunk #{chunk_count}: {preview}")

        # ── 提取音频 base64 数据 ──
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = getattr(choice, "delta", None)
        if not delta:
            continue

        # 从 delta.audio.data 获取 base64 编码的 WAV
        audio_obj = getattr(delta, "audio", None)
        if not audio_obj:
            continue

        b64_data = None
        if isinstance(audio_obj, dict):
            b64_data = audio_obj.get("data")
        elif hasattr(audio_obj, "data"):
            b64_data = audio_obj.data

        if not b64_data:
            continue

        # 解码 base64 -> WAV bytes
        wav_bytes = base64.b64decode(b64_data)
        audio_chunk_count += 1

        # 第一个 chunk: 读取 WAV 参数
        if audio_chunk_count == 1:
            sample_rate, channels, bits_per_sample = _parse_wav_params(wav_bytes)
            if debug:
                print(f"\n  [DEBUG] WAV 参数: {sample_rate}Hz, {channels}ch, {bits_per_sample}bit")
                print(f"  [DEBUG] 第一个 chunk WAV 大小: {len(wav_bytes)} bytes")

        # 剥掉 WAV 头，只保留 PCM 数据
        pcm_data = _extract_pcm_from_wav(wav_bytes)
        pcm_parts.append(pcm_data)

        total_pcm = sum(len(p) for p in pcm_parts)
        print(f"\r  已接收 {audio_chunk_count} 个音频 chunk, PCM 数据: {total_pcm / 1024:.1f} KB", end="", flush=True)

    print(f"\n共接收 {chunk_count} 个 chunk, 其中 {audio_chunk_count} 个含音频数据")

    # ── 合并 PCM 并写入完整 WAV 文件 ──────────────────────────
    if pcm_parts:
        all_pcm = b"".join(pcm_parts)
        wav_header = _build_wav_header(len(all_pcm), sample_rate, channels, bits_per_sample)

        with open(output_path, "wb") as f:
            f.write(wav_header)
            f.write(all_pcm)

        file_size = os.path.getsize(output_path)
        duration = len(all_pcm) / (sample_rate * channels * bits_per_sample // 8)
        print(f"✅ 完成！音频已保存到 {output_path} ({file_size / 1024:.1f} KB, {duration:.1f}s)")
    else:
        print("⚠️  流式模式未收到音频数据，尝试非流式模式...")
        non_stream_tts(text, output_path, voice, audio_format, api_key, base_url, debug)


def non_stream_tts(
    text: str,
    output_path: str = "output.wav",
    voice: str = "mimo_default",
    audio_format: str = "wav",
    api_key: str | None = None,
    base_url: str = "https://api.xiaomimimo.com/v1",
    debug: bool = False,
):
    """非流式 TTS 调用（作为回退方案）"""
    api_key = api_key or os.getenv("MIMO_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base_url)

    print("使用非流式模式...")

    messages = [
        {"role": "user", "content": "请朗读以下内容"},
        {"role": "assistant", "content": text},
    ]

    completion = client.chat.completions.create(
        model="mimo-v2-tts",
        messages=messages,
        audio={
            "format": audio_format,
            "voice": voice,
        },
    )

    resp_json = completion.model_dump_json(indent=2)

    if debug:
        debug_path = output_path + ".debug.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(resp_json)
        print(f"  [DEBUG] 完整响应已保存到 {debug_path}")

    # 尝试提取音频
    audio_data = None
    if completion.choices:
        msg = completion.choices[0].message

        if hasattr(msg, "audio") and msg.audio:
            if isinstance(msg.audio, dict):
                audio_data = msg.audio.get("data")
            elif hasattr(msg.audio, "data"):
                audio_data = msg.audio.data

    if audio_data:
        wav_bytes = base64.b64decode(audio_data)
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        file_size = os.path.getsize(output_path)
        print(f"✅ 完成！音频已保存到 {output_path} ({file_size / 1024:.1f} KB)")
    else:
        print(f"⚠️  未找到音频数据")
        print(f"  响应预览: {resp_json[:500]}...")


# ── CLI 入口 ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MiMo-V2-TTS 流式语音合成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 流式生成并保存为 WAV
  python mimo_v2_tts_stream.py "你好，我是小米语音助手"

  # 带调试输出（查看每个 chunk 结构）
  python mimo_v2_tts_stream.py "你好" --debug

  # 非流式模式
  python mimo_v2_tts_stream.py "你好" --no-stream

  # 指定语音风格
  python mimo_v2_tts_stream.py "你好" --voice mimo_default
        """,
    )
    parser.add_argument("text", nargs="?", help="要合成的文本")
    parser.add_argument("--file", "-f", help="从文件读取文本")
    parser.add_argument("--output", "-o", default="output.wav", help="输出文件路径 (默认: output.wav)")
    parser.add_argument("--voice", "-v", default="mimo_default", help="语音风格 (默认: mimo_default)")
    parser.add_argument("--format", dest="fmt", default="wav", help="音频格式 (默认: wav)")
    parser.add_argument("--no-stream", action="store_true", help="使用非流式模式")
    parser.add_argument("--debug", "-d", action="store_true", help="调试模式：打印 chunk 结构并保存到 .debug.json")
    parser.add_argument("--api-key", help="MiMo API Key (也可用 MIMO_API_KEY 环境变量)")
    parser.add_argument("--base-url", default="https://api.xiaomimimo.com/v1", help="API 端点")

    args = parser.parse_args()

    # 获取文本
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        print("\n错误: 请提供要合成的文本，或使用 --file 指定文本文件。")
        sys.exit(1)

    if not text:
        print("错误: 文本内容为空。")
        sys.exit(1)

    # 执行
    if args.no_stream:
        non_stream_tts(
            text=text,
            output_path=args.output,
            voice=args.voice,
            audio_format=args.fmt,
            api_key=args.api_key,
            base_url=args.base_url,
            debug=args.debug,
        )
    else:
        stream_tts(
            text=text,
            output_path=args.output,
            voice=args.voice,
            audio_format=args.fmt,
            api_key=args.api_key,
            base_url=args.base_url,
            debug=args.debug,
        )


if __name__ == "__main__":
    main()
