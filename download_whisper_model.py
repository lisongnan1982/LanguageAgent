#!/usr/bin/env python3
"""
下载 Whisper 模型文件到本地服务器
使用方法: python download_whisper_model.py
"""

import os
import requests
from pathlib import Path

# 模型配置
MODEL_ID = "Xenova/whisper-tiny"
BASE_URL = "https://huggingface.co"
# 如果 huggingface.co 访问不了，可以改用镜像：
# BASE_URL = "https://hf-mirror.com"

# 需要下载的文件
FILES = [
    "config.json",
    "generation_config.json", 
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "onnx/encoder_model_quantized.onnx",
    "onnx/decoder_model_merged_quantized.onnx",
]

def download_file(url, dest_path):
    """下载文件"""
    print(f"下载: {url}")
    print(f"  -> {dest_path}")
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                print(f"\r  进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
    print("\n  完成!")

def main():
    # 目标目录
    output_dir = Path("models") / MODEL_ID
    print(f"模型将下载到: {output_dir.absolute()}")
    print(f"模型来源: {BASE_URL}")
    print("-" * 50)
    
    for file_path in FILES:
        url = f"{BASE_URL}/{MODEL_ID}/resolve/main/{file_path}"
        dest = output_dir / file_path
        
        if dest.exists():
            print(f"跳过 (已存在): {file_path}")
            continue
            
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  错误: {e}")
            print(f"  提示: 如果无法访问 huggingface.co，请修改脚本中的 BASE_URL 为镜像地址")
    
    print("-" * 50)
    print("下载完成!")
    print(f"模型目录: {output_dir.absolute()}")
    print("\n请确保服务器能够访问 /models/ 路径")

if __name__ == "__main__":
    main()

