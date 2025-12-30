# -*- coding: utf-8 -*-
"""
Entry point for the robust_ocm.eval module.

This allows the module to be run as:
    python -m robust_ocm.eval --gt <gt_file> --pred <pred_dir>
"""

from .cli import main

if __name__ == "__main__":
    exit(main())