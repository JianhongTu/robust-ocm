#!/bin/bash

# Get CUDA version from command line argument
CUDA_VERSION=${1:-"12.6"}

echo "Installing PaddlePaddle for CUDA $CUDA_VERSION..."

# Determine the correct package index based on CUDA version
case $CUDA_VERSION in
    "12.6")
        PACKAGE_INDEX="cu126"
        ;;
    "13.0")
        PACKAGE_INDEX="cu130"
        ;;
    *)
        echo "Error: Unsupported CUDA version $CUDA_VERSION"
        echo "Supported versions: 12.6, 13.0"
        exit 1
        ;;
esac

echo "Using package index: $PACKAGE_INDEX"

# Install PaddlePaddle GPU
python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/$PACKAGE_INDEX/

# Install PaddleOCR
python -m pip install paddleocr

echo "Installation completed!"