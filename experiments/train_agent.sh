#!/usr/bin/env bash
set -euo pipefail

python experiments/train_agent.py \
  --agent_type aec \
  --seed 779 \
  --start_step 20 \
  --test_env_num 32 \
  --num_frames 70000 \
  --model_name llmabs \
  --eval_frequency 10
