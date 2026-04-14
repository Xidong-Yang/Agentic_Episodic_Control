#!/usr/bin/env bash
set -euo pipefail

python experiments/test_agent.py \
  --agent_type aec \
  --seed 2 \
  --start_step 0 \
  --test_env_num 100 \
  --num_frames 100000 \
  --model_name llmabs \
  --eval_frequency 10
