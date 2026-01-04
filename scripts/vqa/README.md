# VQA Scripts

This directory contains scripts for Visual Question Answering (VQA) tasks.

## NIAH Inference Script

The `niah_inference.py` script evaluates vision-language models on the NIAH (Needle in a Haystack) task, where models must answer questions about "secrets" hidden in long documents rendered as images.

### Usage

```bash
micromamba run -n test python scripts/vqa/niah_inference.py \
    --data data/niah/data.json \
    --images data/niah/longbenchv2_img/images \
    --output results/niah/model_name.json \
    --config config/vqa/qwen3-vl.json \
    --base_url http://localhost:8000/v1 \
    --max_workers 128
```

### Arguments

- `--data, -d`: Path to NIAH data.json file
- `--images, -i`: Directory containing rendered images
- `--output, -o`: Output JSON file for results
- `--config, -c`: Path to model configuration JSON file
- `--base_url`: API base URL (default: http://localhost:8000/v1)
- `--api_key`: API key (optional for local vLLM)
- `--max_workers`: Number of concurrent workers (default: 128)
- `--presence_penalty`: Presence penalty for repetition control (default: 0.0)
- `--case_sensitive`: Use case-sensitive answer matching (default: False)
- `--limit`: Limit processing to N instances for testing (optional)

### Model Configuration

Model configurations are stored in `config/vqa/`. Each config file should contain:

```json
{
  "model_name": "Qwen/Qwen3-VL-8B-Instruct",
  "prompt_template": "{question}",
  "max_tokens": 128,
  "temperature": 0.0,
  "timeout": 60,
  "extra_body": null
}
```

Available configs:
- `qwen3-vl.json`: Qwen3-VL-8B-Instruct
- `internvl-3.5.json`: InternVL3.5-8B
- `kimi-vl.json`: Kimi-VL-A3B
- `glm.json`: GLM-4.1V-9B-Thinking

### Output Format

The script outputs a JSON file with:
- Overall accuracy statistics
- Per-instance accuracy breakdown
- Detailed results for each question including:
  - Question text
  - Ground truth answer
  - Model output
  - Correctness evaluation
  - Error information (if any)

### Example Output

```json
{
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "config": "config/vqa/qwen3-vl.json",
  "total_questions": 610,
  "correct": 550,
  "accuracy": 90.16,
  "errors": 2,
  "instance_accuracies": {
    "66ebed525a08c7b9b35e1cb4": {
      "total": 5,
      "correct": 5,
      "accuracy": 100.0
    },
    ...
  },
  "results": [...]
}
```

### Evaluation Metrics

- **Accuracy**: Percentage of questions answered correctly
- **Per-instance accuracy**: Accuracy for each document instance
- **Error rate**: Percentage of API/processing errors

The script checks if the model's output **contains** the ground truth answer (case-insensitive by default). Use `--case_sensitive` for exact case matching.
