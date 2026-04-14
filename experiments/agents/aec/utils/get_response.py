import os
import json
from openai import OpenAI
from ..config import api_key, base_url

client = OpenAI(api_key=api_key, base_url=base_url)


def get_response(messages):
    """Get a response from the remote LLM API."""
    response = client.chat.completions.create(
        model="deepseek-r1-distill-qwen-32b",
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content


def save_work_mem_to_json(saving_agent_logs_path, full_work_mem_prompt, response, env_idx):
    """Save work memory prompt and response to a JSON file."""
    data = {
        "prompt": full_work_mem_prompt,
        "response": response,
    }
    os.makedirs(saving_agent_logs_path, exist_ok=True)
    filename = os.path.join(saving_agent_logs_path, f"work_mem_env_{env_idx}.json")
    with open(filename, 'a') as f:
        json.dump(data, f, indent=4)
