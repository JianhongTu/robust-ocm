#!/bin/bash

# Script to evaluate OCR predictions for multiple subdirectories
# Usage: ./eval_ocr.sh <pred_base_dir>
# Example: ./eval_ocr.sh data/pred/ds

# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <pred_base_dir>"
#     echo "Example: $0 data/pred/ds"
#     exit 1
# fi

model_suffix="mineru"
pred_base_dir="data/pred/${model_suffix}"
results_base="results/${model_suffix}"

# Create results directory if it doesn't exist
mkdir -p "$results_base"

# Loop over each subdirectory in pred_base_dir
for subdir in "$pred_base_dir"/*/; do
    # Remove trailing slash
    subdir=${subdir%/}
    # Get the basename
    subdir_name=$(basename "$subdir")
    
    # Determine the ground truth file based on subdir_name
    if [[ "$subdir_name" =~ fontsize_([0-9]+) ]]; then
        fontsize="${BASH_REMATCH[1]}"
        gt_file="data/ocr1/adv_fontsize_${fontsize}/text_ground_truth.json"
    else
        gt_file="data/ocr1/longbenchv2_img/text_ground_truth.json"
    fi
    
    if [ -d "$subdir" ]; then
        pred_dir="$subdir"
        output_file="$results_base/${subdir_name}_evaluation.json"
        
        # Skip if result file already exists
        if [ -f "$output_file" ]; then
            echo "Skipping $subdir_name (result file already exists)"
            continue
        fi
        
        echo "Evaluating subdirectory: $subdir_name"
        echo "Prediction dir: $pred_dir"
        echo "Output file: $output_file"
        
        # Run the evaluation
        micromamba run -n robust-ocm python -m robust_ocm.eval \
            --gt "$gt_file" \
            --pred "$pred_dir" \
            --output "$output_file"
        
        echo "Completed evaluation for $subdir_name"
        echo
    fi
done

echo "All evaluations completed. Results saved in $results_base"
