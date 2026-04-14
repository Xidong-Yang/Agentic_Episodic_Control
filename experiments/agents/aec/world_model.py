import logging
import random
import re

from .utils.prompts import get_world_model_action_prompt, get_world_model_update_prompt

logger = logging.getLogger(__name__)


def update_world_models(llm_backend, previous_world_models, previous_observations,
                        actions, new_observations):
    """Update world models using LLM-based knowledge graph construction."""
    messages = [
        get_world_model_update_prompt(wm, prev_obs, act, new_obs)
        for wm, prev_obs, act, new_obs
        in zip(previous_world_models, previous_observations, actions, new_observations)
    ]
    return llm_backend.generate(messages)


def get_action_from_world_model(llm_backend, state_list, history_list,
                                world_model_list, env_description, env_missions,
                                list_actions, encoded_actions):
    """Use the world model + LLM to select an action for each environment."""
    messages = [
        get_world_model_action_prompt(obs, history, wm, env_description, mission)
        for obs, history, wm, mission
        in zip(state_list, history_list, world_model_list, env_missions)
    ]
    texts, tokens_per_env, llm_calls_per_env = llm_backend.generate(messages)

    llm_actions = []
    for text in texts:
        try:
            action_text = text.strip().split('<action>')[1].split('</action>')[0]
            action_text = action_text.lower().strip()
            if action_text == 'pickup':
                action_text = 'pick up'
            action_idx = get_action_index(action_text, list_actions)
        except (IndexError, KeyError, AttributeError):
            logger.warning("Failed to parse action, using random: %s", text)
            action_idx = random.randint(0, len(encoded_actions) - 1)
        if action_idx not in range(len(encoded_actions)):
            logger.warning("Action out of range, using random: %s", action_idx)
            action_idx = random.randint(0, len(encoded_actions) - 1)
        llm_actions.append(action_idx)

    return llm_actions, tokens_per_env, llm_calls_per_env


def get_action_index(llm_action, list_actions):
    """Match LLM action string to an action index."""
    llm_action = llm_action.lower().strip()

    for i, action in enumerate(list_actions):
        if llm_action == action.lower():
            return i

    for i, action in enumerate(list_actions):
        if llm_action in action.lower() or action.lower() in llm_action:
            return i

    match = re.search(r'\d+', llm_action)
    if match:
        index = int(match.group())
        if 0 <= index < len(list_actions):
            return index

    return random.randint(0, len(list_actions) - 1)
