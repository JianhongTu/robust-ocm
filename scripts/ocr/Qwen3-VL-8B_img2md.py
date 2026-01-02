#!/usr/bin/env python3
"""
Qwen3-VL-8B OCR inference script with concurrent processing.

To start vLLM server with data parallelism for Qwen3-VL-8B:

# Single GPU:
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --trust-remote-code

# Multiple GPUs (data parallelism):
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 4 \
    --dtype auto \
    --trust-remote-code

Then run this script:
micromamba run -n test python ./scripts/ocr/Qwen3-VL-8B_img2md.py \
    --input data/longbenchv2_img/images \
    --output data/pred/qwenvl \
    --base_url http://localhost:8000/v1 \
    --model_name Qwen/Qwen3-VL-8B-Instruct \
    --max_workers 32
"""

from openai import OpenAI, APIConnectionError
import base64
import os
import time
import sys
import argparse
import concurrent.futures
from tqdm import tqdm

def encode_image(image_path):
    """
    Encode the image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

prompt = r"""You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:
1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \[ \]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
"""

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
            model=model_name,
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
            max_tokens=8192,
            timeout=1000,
            presence_penalty=presence_penalty,
        )

        result = response.choices[0].message.content

        with open(output_path, "w", encoding='utf-8') as f:
            print(result, file=f)

        return f"✓ 成功处理: {image_file}"
    except APIConnectionError as e:
        return f"✗ 连接超时: {image_file}, 错误: {str(e)}"
    except Exception as e:
        return f"✗ 处理失败: {image_file}, 错误: {str(e)}"


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Qwen3-VL OCR inference with concurrent processing')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing images')
    
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for OCR results')
    
    parser.add_argument('--base_url', type=str,
                       default='http://localhost:8000/v1',
                       help='API base URL')
    
    parser.add_argument('--api_key', type=str,
                       default=None,
                       help='API key (optional for local vLLM)')
    
    parser.add_argument('--model_name', type=str,
                       default='Qwen/Qwen3-VL-8B-Instruct',
                       help='Model name')
    
    parser.add_argument('--max_workers', type=int,
                       default=32,
                       help='Number of concurrent workers')
    
    parser.add_argument('--presence_penalty', type=float,
                       default=0.0,
                       help='Presence penalty for repetition control (0.0 to 2.0)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    image_dir = args.input
    result_dir = args.output
    os.makedirs(result_dir, exist_ok=True)

    # 创建OpenAI客户端
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key if args.api_key else "dummy",
    )

    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = [f for f in os.listdir(image_dir)
                  if os.path.isfile(os.path.join(image_dir, f)) and
                  any(f.lower().endswith(ext) for ext in image_extensions)]
    
    # 检查已存在的文件
    existing_files = []
    new_files = []
    for image_file in image_files:
        output_path = os.path.join(result_dir, os.path.splitext(image_file)[0] + ".md")
        if os.path.exists(output_path):
            existing_files.append(image_file)
        else:
            new_files.append(image_file)
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"其中 {len(existing_files)} 个已处理，{len(new_files)} 个待处理")
    
    if len(new_files) == 0:
        print("所有文件都已处理完成！")
        sys.exit(0)
    
    print(f"开始并发处理 (max_workers={args.max_workers})...")
    
    # 使用线程池并发处理
    completed_count = 0
    failed_count = 0
    skipped_count = len(existing_files)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_image, client, image_file, image_dir, result_dir, args.model_name, args.presence_penalty): image_file
            for image_file in new_files
        }
        
        # 使用tqdm显示进度条
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(new_files), desc="处理图片"):
            try:
                result = future.result()
                results.append(result)
                if "✓ 成功处理" in result:
                    completed_count += 1
                elif "✗" in result:
                    failed_count += 1
                elif "⏭ 跳过已存在" in result:
                    skipped_count += 1
            except Exception as exc:
                failed_count += 1
                results.append(f"✗ 异常: {str(exc)}")
    
    print(f"\n处理完成统计:")
    print(f"✓ 成功处理: {completed_count} 个")
    print(f"⏭ 跳过已存在: {skipped_count} 个")
    print(f"✗ 处理失败: {failed_count} 个")
    print(f"📁 总共: {len(image_files)} 个文件")
    print(f"结果保存在: {result_dir}")
    
    # 如果有失败的任务，打印详细信息
    if failed_count > 0:
        print("\n失败详情:")
        for result in results:
            if "✗" in result:
                print(f"  - {result}")