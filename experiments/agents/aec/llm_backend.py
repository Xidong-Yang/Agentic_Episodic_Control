import logging

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .constants import MAX_MODEL_LEN, MAX_TOKENS

logger = logging.getLogger(__name__)


class LLMBackend:
    """Wrapper around vLLM for local LLM inference."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.llm = LLM(model=model_name, max_model_len=MAX_MODEL_LEN)
        self.sampling_params = SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._is_deepseek = 'deepseek' in model_name.lower()

    def generate(self, messages):
        """Call the LLM with a batch of message lists.

        Returns:
            texts: list of output strings
            tokens_per_env: list of total token counts per input
            llm_calls_per_env: list of 1s (one call per input)
        """
        text_ls = [
            self.tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            for msg in messages
        ]
        outputs = self.llm.generate(text_ls, self.sampling_params)

        texts = []
        tokens_per_env = []
        for output in outputs:
            raw_text = output.outputs[0].text
            if self._is_deepseek:
                raw_text = raw_text.split('</think>')[1].strip()
            texts.append(raw_text)
            tokens_per_env.append(
                len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            )

        llm_calls_per_env = [1] * len(messages)
        return texts, tokens_per_env, llm_calls_per_env
