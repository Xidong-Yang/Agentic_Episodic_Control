import logging

from .utils.prompts import (
    get_critical_prompt,
    get_llm_abstruct_prompt,
    get_llm_abstruct_prompt1,
    get_llm_abstruct_prompt2,
)

logger = logging.getLogger(__name__)


def llm_abstract_obs(llm_backend, prompt_obs, env_description):
    """Generate LLM abstractions of observations (single-step)."""
    messages = [get_llm_abstruct_prompt(obs, env_description) for obs in prompt_obs]
    texts, tokens_per_env, llm_calls_per_env = llm_backend.generate(messages)

    results = []
    for original_prompt, text in zip(prompt_obs, texts):
        try:
            results.append(original_prompt + f"\nLLM abstract: {text}")
        except (IndexError, AttributeError):
            logger.warning("LLM abstract parse failed, using original prompt")
            results.append(original_prompt + f"\nLLM abstract: {original_prompt}")

    return results, tokens_per_env, llm_calls_per_env


def llm_abstract_obs_two_step(llm_backend, prompt_obs, env_description):
    """Two-step LLM abstraction with obstacle detection."""
    messages_1 = [get_llm_abstruct_prompt1(obs, env_description) for obs in prompt_obs]
    texts_1, tokens_1, _ = llm_backend.generate(messages_1)

    messages_2 = [get_llm_abstruct_prompt2(obs, env_description) for obs in prompt_obs]
    texts_2, tokens_2, _ = llm_backend.generate(messages_2)

    tokens_per_env = [t1 + t2 for t1, t2 in zip(tokens_1, tokens_2)]

    results = []
    for original_prompt, text1, text2 in zip(prompt_obs, texts_1, texts_2):
        try:
            results.append(original_prompt + f"\nLLM abstract: {text1}\n Obstacle: {text2}")
        except (IndexError, AttributeError):
            logger.warning("LLM abstract parse failed, using original prompt")
            results.append(original_prompt + f"\nLLM abstract: {original_prompt}")

    llm_calls_per_env = [1] * len(prompt_obs)
    return results, tokens_per_env, llm_calls_per_env


def is_critical_state(llm_backend, state_prompts, env_missions, env_description,
                      split_fn, extract_raw_fn, check_mode='llmabs'):
    """Determine if states are critical using LLM evaluation."""
    obs_raw = []
    obs_llm_abs = []
    for prompt in state_prompts:
        _, llm_abstract = split_fn(prompt)
        obs_raw.append(extract_raw_fn(prompt))
        obs_llm_abs.append(llm_abstract)

    if check_mode.lower() == 'llmabs' and obs_llm_abs:
        return get_critical_state(llm_backend, obs_llm_abs, env_description, env_missions)
    return get_critical_state(llm_backend, obs_raw, env_description, env_missions)


def get_critical_state(llm_backend, prompt_obs, env_description, env_missions):
    """Use the LLM to evaluate whether observations represent critical states."""
    messages = [
        get_critical_prompt(prompt, env_description, env_mission)
        for prompt, env_mission in zip(prompt_obs, env_missions)
    ]
    texts, _, _ = llm_backend.generate(messages)

    results = []
    for text in texts:
        try:
            results.append(1 if 'yes' in text.lower() else 0)
        except (IndexError, AttributeError):
            logger.warning("Critical state parse failed, defaulting to critical")
            results.append(1)
    return results
