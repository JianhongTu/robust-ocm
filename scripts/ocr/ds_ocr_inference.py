#!/usr/bin/env python3
"""
DeepSeek-OCR inference script with concurrent processing.

To start vLLM server with data parallelism for DeepSeek-OCR:

# Single GPU:
vllm serve deepseek-ai/DeepSeek-OCR \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --trust-remote-code \
    --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0

# Multiple GPUs (data parallelism):
vllm serve deepseek-ai/DeepSeek-OCR \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 4 \
    --dtype auto \
    --trust-remote-code \
    --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0

Then run this script:
micromamba run -n test python scripts/ocr/ds_ocr_inference.py \
    --input data/ocr \
    --output data/pred/ds \
    --base_url http://localhost:8000/v1 \
    --model_name deepseek-ai/DeepSeek-OCR \
    --max_workers 1024
"""

from openai import OpenAI, APIConnectionError
import base64
import os
import time
import sys
import argparse
import concurrent.futures
from tqdm import tqdm
import re
import json
import glob


def encode_image(image_path):
    """
    Encode the image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

prompt = '<image>\nFree OCR.'

def process_image(client, image_file, image_dir, result_dir, model_name, presence_penalty=0.0):
    """
    处理单个图片文件
    """
    try:
        # 检查输出文件是否已存在
        output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
        if os.path.exists(output_path):
            return f"⏭ 跳过已存在: {image_file}"

        image_path = os.path.join(image_dir, image_file)
        base64_image = encode_image(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-OCR",
            messages=[{
                'role':'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt,
                    },
                    {
                        'type': 'image_url',
                        'image_url': {'url': data_url},
                    }
                ],
            }],
            max_tokens=8100,
            temperature=0.0,
            extra_body={
                "skip_special_tokens": False,
                # args used to control custom logits processor
                "vllm_xargs": {
                    "ngram_size": 30,
                    "window_size": 90,
                    # whitelist: <td>, </td>
                    "whitelist_token_ids": [128821, 128822],
                },
            },
        )

        result = response.choices[0].message.content

        # Clean the content
        cleaned_result = clean_ocr_content(result)

        with open(output_path, "w", encoding='utf-8') as f:
            print(cleaned_result, file=f)

        return f"✓ 成功处理: {image_file}"
    except APIConnectionError as e:
        return f"✗ 连接超时: {image_file}, 错误: {str(e)}"
    except Exception as e:
        return f"✗ 处理失败: {image_file}, 错误: {str(e)}"


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='DeepSeek-OCR inference with concurrent processing')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Parent directory containing subdirectories with images folders')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for OCR results')
    
    parser.add_argument('--base_url', type=str,
                       default='http://localhost:8000/v1',
                       help='API base URL')
    
    parser.add_argument('--api_key', type=str,
                       default=None,
                       help='API key (optional for local vLLM)')
    
    parser.add_argument('--model_name', type=str,
                       default='deepseek-ai/DeepSeek-OCR',
                       help='Model name')
    
    parser.add_argument('--max_workers', type=int,
                       default=1024,
                       help='Number of concurrent workers')
    
    parser.add_argument('--presence_penalty', type=float,
                       default=0.0,
                       help='Presence penalty for repetition control (0.0 to 2.0)')
    
    return parser.parse_args()


def clean_formula(text: str) -> str:
    """Clean formula formatting in OCR output"""
    formula_pattern = r'\\\[(.*?)\\\]'
    
    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'
    
    cleaned_text = re.sub(formula_pattern, process_formula, text)
    return cleaned_text


def re_match(text):
    """
    Extract matches from OCR output

    Args:
        text: OCR output text

    Returns:
        Tuple of (all_matches, filtered_matches)
    """
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    # Extract the full match strings
    filtered_matches = []
    for a_match in matches:
        filtered_matches.append(a_match[0])
    return matches, filtered_matches


def clean_ocr_content(content: str) -> str:
    """
    Clean and normalize OCR output

    Args:
        content: Raw OCR output

    Returns:
        Cleaned content
    """
    # Clean formulas
    content = clean_formula(content)

    # Remove reference and detection tags in a loop
    matches_ref, filtered_matches = re_match(content)
    for a_match in filtered_matches:
        content = content.replace(a_match, '')

    # Normalize line breaks and remove center tags
    content = content.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')

    return content.strip()


if __name__ == "__main__":
    args = parse_args()
    
    parent_input = args.input
    base_output = args.output
    os.makedirs(base_output, exist_ok=True)

    # 创建OpenAI客户端
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key if args.api_key else "dummy",
    )

    # Find subdirs with images
    image_subdirs = []
    for item in os.listdir(parent_input):
        item_path = os.path.join(parent_input, item)
        if os.path.isdir(item_path):
            images_path = os.path.join(item_path, 'images')
            if os.path.isdir(images_path):
                image_subdirs.append(item)

    if not image_subdirs:
        print("No subdirectories with 'images' folder found.")
        sys.exit(1)

    # 第一步：扫描所有子目录，构建统一的任务队列
    print("扫描所有子目录中的图片文件...")
    all_tasks = []  # List of (image_file, image_dir, result_dir, subdir_name)
    subdir_stats = {}  # Track stats per subdirectory
    total_images = 0
    total_skipped = 0

    for subdir in image_subdirs:
        image_dir = os.path.join(parent_input, subdir, 'images')
        result_dir = os.path.join(base_output, subdir)
        os.makedirs(result_dir, exist_ok=True)

        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        image_files = [f for f in os.listdir(image_dir)
                      if os.path.isfile(os.path.join(image_dir, f)) and
                      any(f.lower().endswith(ext) for ext in image_extensions)]
        
        total_images += len(image_files)

        # 检查已存在的文件，只添加新文件到任务队列
        existing_count = 0
        new_count = 0
        for image_file in image_files:
            output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
            if os.path.exists(output_path):
                existing_count += 1
                total_skipped += 1
            else:
                all_tasks.append((image_file, image_dir, result_dir, subdir))
                new_count += 1

        subdir_stats[subdir] = {
            'total': len(image_files),
            'existing': existing_count,
            'new': new_count,
            'completed': 0,
            'failed': 0
        }
        
        print(f"子目录 {subdir}: {len(image_files)} 个图片，{existing_count} 已处理，{new_count} 待处理")

    if len(all_tasks) == 0:
        print("所有文件都已处理完成！")
        sys.exit(0)

    print(f"\n总共待处理: {len(all_tasks)} 个文件，已跳过: {total_skipped} 个文件")
    print(f"开始并发处理 (max_workers={args.max_workers})...\n")
    
    # 第二步：一次性提交所有任务到线程池
    total_completed = 0
    total_failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_image, client, image_file, image_dir, result_dir, args.model_name, args.presence_penalty): (image_file, subdir)
            for image_file, image_dir, result_dir, subdir in all_tasks
        }
        
        # 使用tqdm显示总体进度条
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_tasks), desc="总体进度"):
            try:
                result = future.result()
                image_file, subdir = futures[future]
                results.append((result, subdir))
                
                if "✓ 成功处理" in result:
                    total_completed += 1
                    subdir_stats[subdir]['completed'] += 1
                elif "✗" in result:
                    total_failed += 1
                    subdir_stats[subdir]['failed'] += 1
            except Exception as exc:
                total_failed += 1
                image_file, subdir = futures[future]
                subdir_stats[subdir]['failed'] += 1
                results.append((f"✗ 异常: {str(exc)}", subdir))
    
    # 第三步：打印详细统计信息
    print(f"\n处理完成统计 (按子目录):")
    print("-" * 80)
    for subdir in image_subdirs:
        stats = subdir_stats[subdir]
        print(f"{subdir:30} | 总: {stats['total']:4} | 已: {stats['existing']:4} | 成: {stats['completed']:4} | 失: {stats['failed']:4}")
    print("-" * 80)
    print(f"{'总计':30} | 总: {total_images:4} | 已: {total_skipped:4} | 成: {total_completed:4} | 失: {total_failed:4}")
    
    print(f"\n结果保存在: {base_output}")
    
    # 如果有失败的任务，打印详细信息
    if total_failed > 0:
        print(f"\n失败详情 (共 {total_failed} 个):")
        for result, subdir in results:
            if "✗" in result:
                print(f"  [{subdir}] {result}")