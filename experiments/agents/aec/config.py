import os


api_key = os.getenv("AEC_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
base_url = os.getenv("AEC_API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
