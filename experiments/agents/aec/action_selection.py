import logging
import random

import numpy as np

from .constants import argmax_with_random_tiebreak
from .observation import is_critical_state, llm_abstract_obs
from .world_model import get_action_from_world_model

logger = logging.getLogger(__name__)


def retrieve_ec_values(embedding, ec_buffer, n_actions, track_peeks=False, peek_counter=None):
    """Query episodic memory for Q-values across all actions.

    Args:
        peek_counter: dict with 'total' and 'non_null' keys, mutated in place when track_peeks=True
    """
    act_values = []
    for j in range(n_actions):
        if not ec_buffer[j].build_tree:
            continue
        qd, _, _, _ = ec_buffer[j].peek(embedding, None, modify=False, get_act=True)
        if track_peeks and peek_counter is not None:
            peek_counter['total'] += 1
            if qd is not None:
                peek_counter['non_null'] += 1
        if qd is None:
            qd = 0
        act_values.append(qd)
    return act_values


def select_actions(agent, states_prompts, poss_acts, step, model='raw',
                   max_epsilon=1.0, min_epsilon=0.1, decay_steps=500):
    """Select actions during training using epsilon-greedy + EC memory + world model."""
    use_critical = 'wocritical' not in model.lower()
    n_envs = agent.n_envs

    epsilon = max(min_epsilon, max_epsilon - (step / decay_steps) * (max_epsilon - min_epsilon))
    act_values = [[] for _ in range(n_envs)]
    act_idxs = [None] * n_envs

    for i in range(n_envs):
        if random.random() < epsilon:
            act_idxs[i] = random.randint(0, len(poss_acts) - 1)

    active_envs = [i for i in range(n_envs) if act_idxs[i] is None]

    if active_envs:
        active_prompts = [states_prompts[i] for i in active_envs]
        active_missions = [agent.env_mission[i] for i in active_envs]
        if use_critical:
            critical_flags = is_critical_state(
                agent.llm_backend, active_prompts, active_missions,
                agent.env_description, agent._split_prompt_sections,
                agent._extract_raw_observation,
            )
        else:
            critical_flags = [1] * len(active_envs)
    else:
        critical_flags = []

    critical_env_index = [
        idx for flag, idx in zip(critical_flags, active_envs) if flag == 1
    ]
    logger.info("critical env nums: %d", len(critical_env_index))

    if critical_env_index and 'llmabs' in model.lower():
        for i in critical_env_index:
            if act_idxs[i] is not None:
                continue
            _, llm_abstract = agent._split_prompt_sections(states_prompts[i])
            obs_repr = f'LLM abstract:{llm_abstract}' if llm_abstract else ''
            embedding = agent.sentece_transformer_model.encode(obs_repr, show_progress_bar=False)
            values = retrieve_ec_values(embedding, agent.ec_buffer_LLMabs, len(poss_acts))
            if values:
                act_values[i] = values
                act_idxs[i] = argmax_with_random_tiebreak(np.array(values))

    # World model fallback
    obs_raw_list, none_indices, wm_prompts, wm_list, none_missions = [], [], [], [], []
    for i in range(n_envs):
        if act_idxs[i] is None:
            obs_raw_list.append(agent._extract_raw_observation(states_prompts[i]))
            none_indices.append(i)
            wm_prompts.append(agent.work_mem_prompts[i])
            wm_list.append(agent.world_models[i])
            none_missions.append(agent.env_mission[i])

    if obs_raw_list:
        wm_actions, _, _ = get_action_from_world_model(
            agent.llm_backend, obs_raw_list, wm_prompts, wm_list,
            agent.env_description, none_missions,
            agent.list_actions, agent.encoded_actions,
        )
        for idx, action in zip(none_indices, wm_actions):
            act_idxs[idx] = action

    act_ids = [
        poss_acts[idx if idx is not None else random.randint(0, len(poss_acts) - 1)]
        for idx in act_idxs
    ]
    return act_ids, act_idxs, act_values


def select_test_actions(agent, states_prompts, poss_acts, model, envs_done,
                        ec_buffer_raw, ec_buffer_LLMabs, work_mem_prompts,
                        world_models, env_missions, peek_counter):
    """Select actions during testing with token/call tracking."""
    use_critical = 'wocritical' not in model.lower()
    n = len(states_prompts)

    tokens_per_env = {i: 0 for i in range(n)}
    llm_calls_per_env = {i: 0 for i in range(n)}
    step_decision_count = 0
    step_critical_count = 0

    act_idxs = []
    active_envs = []
    act_values = [[] for _ in range(n)]

    for i in range(n):
        if envs_done[i] == 1:
            act_idxs.append(random.randint(0, len(poss_acts) - 1))
        else:
            act_idxs.append(None)
            active_envs.append(i)

    step_decision_count = len(active_envs)

    if active_envs:
        active_prompts = [states_prompts[i] for i in active_envs]
        active_missions = [env_missions[i] for i in active_envs]

        abstracted = active_prompts
        if 'llmabs' in model.lower():
            abstracted, abstract_tokens, abstract_calls = llm_abstract_obs(
                agent.llm_backend, active_prompts, agent.env_description,
            )
            for idx, env_idx in enumerate(active_envs):
                tokens_per_env[env_idx] += abstract_tokens[idx]
                llm_calls_per_env[env_idx] += abstract_calls[idx]

        if use_critical:
            critical_flags = is_critical_state(
                agent.llm_backend, abstracted, active_missions,
                agent.env_description, agent._split_prompt_sections,
                agent._extract_raw_observation,
            )
        else:
            critical_flags = [1] * len(active_envs)

        critical_idxs = []
        critical_prompts = []
        for flag, env_idx, prompt in zip(critical_flags, active_envs, abstracted):
            if flag == 1:
                critical_idxs.append(env_idx)
                critical_prompts.append(prompt)

        step_critical_count = len(critical_prompts)
        logger.info("critical state nums: %d", len(critical_prompts))

        if 'llmabs' in model.lower():
            for i, prompt in zip(critical_idxs, critical_prompts):
                state = agent._extract_llm_abstract_section(prompt)
                emb = agent.sentece_transformer_model.encode(state, show_progress_bar=False)
                values = retrieve_ec_values(emb, ec_buffer_LLMabs, len(poss_acts),
                                            track_peeks=True, peek_counter=peek_counter)
                if values:
                    act_values[i] = values
                    act_idxs[i] = argmax_with_random_tiebreak(np.array(values))
        elif 'raw' in model.lower():
            for i in critical_idxs:
                emb = agent.sentece_transformer_model.encode(states_prompts[i], show_progress_bar=False)
                values = retrieve_ec_values(emb, ec_buffer_raw, len(poss_acts),
                                            track_peeks=True, peek_counter=peek_counter)
                if values:
                    act_values[i] = values
                    act_idxs[i] = argmax_with_random_tiebreak(np.array(values))

    # World model fallback
    obs_raw_list, none_indices, wm_prompts_list = [], [], []
    wm_list, none_missions = [], []
    for i in range(agent.args.test_env_num):
        if envs_done[i] == 1 or act_idxs[i] is not None:
            continue
        obs_raw_list.append(agent._extract_raw_observation(states_prompts[i]))
        none_indices.append(i)
        wm_prompts_list.append(work_mem_prompts[i])
        wm_list.append(world_models[i])
        none_missions.append(env_missions[i])

    if obs_raw_list:
        wm_actions, action_tokens, action_calls = get_action_from_world_model(
            agent.llm_backend, obs_raw_list, wm_prompts_list, wm_list,
            agent.env_description, none_missions,
            agent.list_actions, agent.encoded_actions,
        )
        for idx, action, tokens, calls in zip(none_indices, wm_actions, action_tokens, action_calls):
            act_idxs[idx] = action
            tokens_per_env[idx] += tokens
            llm_calls_per_env[idx] += calls

    for i in range(len(act_idxs)):
        if act_idxs[i] is None:
            act_idxs[i] = random.randint(0, len(poss_acts) - 1)

    act_ids = [poss_acts[idx] for idx in act_idxs]
    return (act_ids, act_idxs, act_values, tokens_per_env,
            llm_calls_per_env, step_decision_count, step_critical_count)
