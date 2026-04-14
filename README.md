# Agentic Episodic Control (AEC)

This repository implements Agentic Episodic Control for BabyAI-based reinforcement learning experiments, integrating episodic memory mechanisms with Large Language Models.

## Overview

This project explores the integration of episodic memory and large language models to enhance reinforcement learning in text-based environments, specifically using the BabyAI platform. The supported workflow centers on:

- `experiments/train_agent.py` for training
- `experiments/test_agent.py` for evaluation
- `experiments/agents/aec/` for the agent implementation

## Features

- AEC agent implementation enhanced with LLM reasoning
- BabyAI text-based environment support
- Training and evaluation entrypoints with shared runtime helpers
- WandB integration for experiment tracking
- Repository-relative defaults

## Project Structure

```text
.
├── experiments/
│   ├── agents/
│   │   ├── base_agent.py
│   │   └── aec/
│   ├── arguments.py
│   ├── runtime.py
│   ├── train_agent.py
│   ├── test_agent.py
│   ├── train_agent.sh
│   └── test_agent.sh
├── babyai-text/
├── LLM_models/
├── models/
├── storage/
├── requirements.txt
└── README.md
```

## Installation

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Install BabyAI text environment dependencies:

```bash
cd babyai-text/babyai
pip install -e .
cd ../gym-minigrid
pip install -e .
cd ..
pip install -e .
```

3. Download the sentence embedding model weights:

```bash
cd experiments/agents/aec/SentenceEmbeddingsModels/bert-sentence-transformers
# Download from HuggingFace: sentence-transformers/all-MiniLM-L6-v2
# Place pytorch_model.bin and model.safetensors in this directory
```

4. Set up the LLM model (e.g., Qwen2.5-32B-Instruct) under `LLM_models/` or specify a custom path via `--llm_name`.

## Configuration

Default paths are resolved relative to the repository.

If you need API-based LLM access, configure credentials through environment variables:

```bash
export AEC_API_KEY=your_api_key
export AEC_API_BASE_URL=your_base_url
```

Compatible fallbacks:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

## Usage

### Train

```bash
bash experiments/train_agent.sh
```

Or run directly:

```bash
python experiments/train_agent.py --agent_type aec --num_frames 70000
```

### Test

```bash
bash experiments/test_agent.sh
```

Or run directly:

```bash
python experiments/test_agent.py --agent_type aec --test_env_num 100
```

## Acknowledgements

This project builds upon:

- BabyAI platform for instruction following and language grounding
- Neural Episodic Control algorithm
- LLM tooling used for text understanding and generation
