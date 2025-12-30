from openai import OpenAI
import json
import os
import requests
import base64
import concurrent.futures
import argparse
from tqdm import tqdm  # 用于显示进度条

def encode_image(image_path):
    """将本地图片转换为base64编码"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_single_image(image_info, prompt_text, api_url, api_key, output_dir, model_name):
    """处理单张图片的函数"""
    image_file, image_dir = image_info
    image_path = os.path.join(image_dir, image_file)

    # 构建请求数据
    data = {
        "model": model_name,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        }
                    }
                ]
            }
        ]
    }

    # 发送请求到本地vLLM API
    headers = {
        'Content-Type': 'application/json',
    }

    # 添加Authorization header如果提供了api_key
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        # 创建Markdown文件名（使用图片名，但扩展名改为.md）
        base_name = os.path.basename(image_file)[:-4]
        md_filename = f"{base_name}.md"
        md_path = os.path.join(output_dir, md_filename)

        # 写入Markdown文件
        with open(md_path, 'w', encoding='utf-8') as md_file:
            md_file.write(f"{content}")

        return f"成功处理: {image_file}"
    except Exception as e:
        return f"处理失败 {image_file}: {str(e)}"

def process_images(image_dir, prompt_text, api_url, api_key=None, output_dir=None, max_workers=5, model_name="internvl3.5-241b-a28b"):
    """处理目录中的所有图片并为每个图片生成单独的Markdown文件（多线程版本）"""

    # 设置输出目录，默认为图片目录
    if output_dir is None:
        output_dir = image_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    # 获取图片文件列表（支持常见格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = [f for f in os.listdir(image_dir)
                  if os.path.isfile(os.path.join(image_dir, f)) and
                  any(f.lower().endswith(ext) for ext in image_extensions)]

    if not image_files:
        print("指定目录中没有找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    # 准备参数列表
    image_infos = [(img_file, image_dir) for img_file in image_files]

    # 使用线程池并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_image, info, prompt_text, api_url, api_key, output_dir, model_name): info[0]
            for info in image_infos
        }

        # 使用tqdm显示进度条
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(image_files), desc="处理图片"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(f"异常: {str(e)}")

    # 打印处理结果摘要
    print("\n处理完成！结果摘要:")
    success_count = sum(1 for r in results if "成功处理" in r)
    failure_count = len(results) - success_count
    print(f"成功: {success_count}, 失败: {failure_count}")

    # 如果有失败的任务，打印详细信息
    if failure_count > 0:
        print("\n失败详情:")
        for result in results:
            if "失败" in result or "异常" in result:
                print(f"  - {result}")

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InternVL OCR inference with local vLLM backend")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for OCR results")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000/v1/chat/completions", help="vLLM API URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key (optional for local vLLM)")
    parser.add_argument("--model-name", type=str, default="OpenGVLab/InternVL3_5-8B", help="Model name")
    parser.add_argument("--max-workers", type=int, default=32, help="Number of concurrent workers")

    args = parser.parse_args()

    # OCR Prompt
    PROMPT_TEXT = r"""
    You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

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

    # 处理图片
    process_images(
        args.input,
        PROMPT_TEXT,
        args.api_url,
        args.api_key,
        args.output,
        args.max_workers,
        args.model_name
    )