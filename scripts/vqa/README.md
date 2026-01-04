# VQA Scripts

This directory contains scripts for Visual Question Answering (VQA) tasks.

## Multiple Choice Inference Script

The `mc_inference.py` script evaluates vision-language models on multiple choice questions from the LongBench-v2 dataset, where models must answer questions based on long documents rendered as images.

### Usage

```bash
micromamba run -n test python scripts/vqa/mc_inference.py \
    --data-json data/longbenchv2/data.json \
    --images-dir data/vqa \
    --output results/qwen3 \
    --config config/vqa/qwen3-vl.json \
    --base_url http://localhost:8000/v1 \
    --max_workers 128
```

### Arguments

- `--data-json`: Path to data.json file (e.g., data/longbenchv2/data.json)
- `--images-dir`: Parent directory containing subdirectories with images/ folders (e.g., data/vqa)
- `--output, -o`: Output directory for result JSON files
- `--config, -c`: Path to model configuration JSON file
- `--base_url`: API base URL (default: http://localhost:8000/v1)
- `--api_key`: API key (optional for local vLLM)
- `--max_workers`: Number of concurrent workers (default: 128)
- `--presence_penalty`: Presence penalty for repetition control (default: 0.0)
- `--limit`: Limit processing to N instances per subdirectory (optional)

### How It Works

1. Loads data.json from the specified path (e.g., `data/longbenchv2/data.json`)
2. Scans images-dir for subdirectories containing `images/` folders (different perturbations)
3. Processes each subdirectory independently
4. Saves results to `{output}/{subdir}_mc_eval.json`

### Data Structure

The script expects:
```
data/
├── longbenchv2/
│   └── data.json                # Dataset file
└── vqa/
    ├── longbenchv2_img/
    │   └── images/              # Original images
    ├── adv_dpi_56/
    │   └── images/              # Low DPI perturbation
    ├── adv_jpeg_10/
    │   └── images/              # JPEG compression
    └── ...
```

Each entry in `data.json` should contain:
```json
{
  "_id": "66fcffd9bb02136c067c94c5",
  "question": "What is the answer?",
  "choice_A": "First option",
  "choice_B": "Second option",
  "choice_C": "Third option",
  "choice_D": "Fourth option",
  "answer": "B",
  "context": "Long context text...",
  "difficulty": "hard",
  "length": "long"
}
```

### Answer Parsing

The script uses multiple strategies to extract answers (A, B, C, or D):
1. Word boundary match: `"The answer is B"` → `B`
2. Parentheses: `"(C)"` → `C`
3. After keywords: `"Answer: D"` → `D`
4. First character fallback: `"A"` → `A`

### Output Format

The script outputs JSON files with:
- Overall accuracy statistics
- Analysis by difficulty (easy/medium/hard)
- Analysis by length (short/medium/long)
- Analysis by context length (8k-16k, 16k-24k, etc.)
- Detailed results for each question including:
  - Question text and choices
  - Ground truth answer
  - Formatted prompt sent to model
  - Model output
  - Extracted answer
  - Correctness evaluation

### Example Output

```json
{
  "subdirectory": "longbenchv2_img",
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "total_questions": 503,
  "correct": 420,
  "accuracy": 83.5,
  "context_length_analysis": {
    "by_context_length": {
      "8k-16k": {"total": 150, "correct": 130, "accuracy": 86.67},
      "16k-24k": {"total": 200, "correct": 165, "accuracy": 82.5}
    }
  },
  "detailed_analysis": {
    "by_difficulty": {
      "easy": {"total": 100, "correct": 95, "accuracy": 95.0},
      "medium": {"total": 200, "correct": 170, "accuracy": 85.0},
      "hard": {"total": 203, "correct": 155, "accuracy": 76.35}
    },
    "by_length": {...},
    "by_difficulty_and_length": {...}
  },
  "results": [...]
}
```

### Model Configuration

Same as NIAH inference. Optionally add a `prompt_template` field:

```json
{
  "model_name": "Qwen/Qwen3-VL-8B-Instruct",
  "prompt_template": "{question}\n\n{choices}\n\nPlease provide your answer in parentheses, e.g., (A).\n\nAnswer:",
  "max_tokens": 512,
  "temperature": 0.0
}
```

Placeholders:
- `{question}`: The question text
- `{choices}`: Formatted choices (A. ..., B. ..., etc.)

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
