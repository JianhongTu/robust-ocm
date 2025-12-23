---
applyTo: '**'
---
In this codebase, we are creating a new benchmark for optical compression models. The dataset we deal with contains very long outputs. Rule: Never directy prints contents of a dataset file, which will overwhelm the system.

Before running python codes, activate the environment via "micromamba activate <env_name>" or you can use "micromamba run -n <env_name> <command>" to run commands within the environment without activating it.

Environments:
1. robust_ocm: for running the main codebase
2. paddle: for running paddle related code (e.g., repo/paddle/bbox_cli.py)
3. deepseek-ocr: for running deepseek related code