import logging
import random
import time
from collections import deque

import gym
import numpy as np
from PIL import Image
from babyai.paral_env_simple import ParallelEnv

from .constants import WORK_MEM_OBS_MAXLEN, WORK_MEM_ACT_MAXLEN
from .utils.lru_knn import LRUKnn
from .action_selection import select_test_actions
from .world_model import update_world_models

logger = logging.getLogger(__name__)


def run_test(agent, env_num):
    """Run the full test loop and return episode statistics."""
    peek_counter = {'total': 0, 'non_null': 0}

    total_tokens_all_episodes = 0
    tokens_per_episode = []
    current_episode_tokens = [0] * agent.args.test_env_num
    episode_start_times = [time.time()] * agent.args.test_env_num
    episode_durations = []
    current_episode_llm_calls = [0] * agent.args.test_env_num
    llm_calls_per_episode = []
    total_decision_count = 0
    total_critical_count = 0
    current_episode_decisions = [0] * agent.args.test_env_num
    current_episode_criticals = [0.0] * agent.args.test_env_num
    decisions_per_episode = []
    criticals_per_episode = []

    agent.test_times += 1
    list_actions = [a.replace("_", " ") for a in agent.args.action_space]
    test_return_per_episode = []
    image_raw = [[] for _ in range(agent.args.test_env_num)]
    image_gif = [[] for _ in range(agent.args.test_env_num)]
    map_gif = [[] for _ in range(agent.args.test_env_num)]
    agent_path_highlighted_gif = [[] for _ in range(agent.args.test_env_num)]
    agent_positions_all_steps = [[] for _ in range(agent.args.test_env_num)]
    work_mem_obs = [deque([], maxlen=WORK_MEM_OBS_MAXLEN) for _ in range(agent.args.test_env_num)]
    work_mem_act = [deque([], maxlen=WORK_MEM_ACT_MAXLEN) for _ in range(agent.args.test_env_num)]

    env_missions = [None] * agent.args.test_env_num
    world_models = [None] * agent.args.test_env_num
    previous_obs = [None] * agent.args.test_env_num
    previous_action = [None] * agent.args.test_env_num
    new_obs = [None] * agent.args.test_env_num

    envs = []
    max_steps = 0
    for i in range(agent.args.test_env_num):
        env = gym.make(agent.args.env_name_list[env_num])
        env.seed(random.randint(0, 1000) * agent.args.seed + i)
        max_steps = env.max_steps
        envs.append(env)
    envs = ParallelEnv(envs)
    obs, infos = envs.reset()

    n_test_envs = len(obs)
    returns = [0] * n_test_envs
    envs_done = [0] * n_test_envs
    obs_queue = [deque([], maxlen=1) for _ in range(n_test_envs)]
    acts_queue = [deque([], maxlen=2) for _ in range(n_test_envs)]
    work_mem = [deque([], maxlen=10) for _ in range(n_test_envs)]

    for j in range(n_test_envs):
        obs_queue[j].append(infos[j]['descriptions'])
    prompts = [
        agent.generate_prompt(goal=obs[j]['mission'], list_actions=list_actions, deque_obs=obs_queue[j])
        for j in range(n_test_envs)
    ]
    encoded_actions = agent.encode_actions(list_actions)
    ec_prompts = prompts

    for j in range(agent.args.test_env_num):
        workmem_raw = agent._extract_raw_observation(ec_prompts[j])
        work_mem_obs[j].append([workmem_raw])
        new_obs[j] = workmem_raw
        env_missions[j] = obs[j]['mission']

    work_mem_prompts = [
        agent.generate_work_mem(work_mem_obs[j], work_mem_act[j])
        for j in range(n_test_envs)
    ]

    # Initial world model update
    active_indices = [i for i in range(len(envs_done)) if envs_done[i] == 0]
    if active_indices:
        active_wm = [world_models[i] for i in active_indices]
        active_prev = [previous_obs[i] for i in active_indices]
        active_act = [previous_action[i] for i in active_indices]
        active_new = [new_obs[i] for i in active_indices]
        updated_wm, wm_tokens, wm_calls = update_world_models(
            agent.llm_backend, active_wm, active_prev, active_act, active_new,
        )
        for idx, orig_idx in enumerate(active_indices):
            world_models[orig_idx] = updated_wm[idx]
            current_episode_tokens[orig_idx] += wm_tokens[idx]
            current_episode_llm_calls[orig_idx] += wm_calls[idx]

    # Load episodic memory buffers
    ec_buffer_raw = []
    ec_buffer_LLMabs = []
    memory_path = agent.args.test_memory_path or agent.saving_agent_memory_path
    for j in range(agent.action_size):
        ec_buffer_raw.append(LRUKnn(
            agent.args, memory_path, agent.action_size,
            agent.args.ec_buffer_size, agent.args.ec_latent_dim,
            env_name=agent.args.name_environment,
            save_qa_path=agent.ec_save_path, save_qa_index=j,
            save_type='raw', test=True,
        ))
    for j in range(agent.action_size):
        ec_buffer_LLMabs.append(LRUKnn(
            agent.args, memory_path, agent.action_size,
            agent.args.ec_buffer_size, agent.args.ec_latent_dim,
            env_name=agent.args.name_environment,
            save_qa_path=agent.ec_save_path, save_qa_index=j,
            save_type='LLMabs', test=True,
        ))
        ec_buffer_LLMabs[j].load(j)
        ec_buffer_raw[j].load(j)

    # Main test loop
    for i in range(max_steps):
        if i % 10 == 0:
            logger.info("Test step %d/%d (%.1f%%)", i + 1, max_steps, (i + 1) / max_steps * 100)

        (action_ids, action_idxs, _, step_tokens, step_calls,
         step_decisions, step_criticals) = select_test_actions(
            agent, ec_prompts, encoded_actions, model=agent.args.model_name,
            envs_done=envs_done, ec_buffer_raw=ec_buffer_raw,
            ec_buffer_LLMabs=ec_buffer_LLMabs,
            work_mem_prompts=work_mem_prompts, world_models=world_models,
            env_missions=env_missions, peek_counter=peek_counter,
        )

        for env_idx, tokens in step_tokens.items():
            current_episode_tokens[env_idx] += tokens
        for env_idx, calls in step_calls.items():
            current_episode_llm_calls[env_idx] += calls

        for env_idx in range(agent.args.test_env_num):
            if envs_done[env_idx] == 0:
                current_episode_decisions[env_idx] += 1
        total_decision_count += step_decisions
        total_critical_count += step_criticals

        active_count = sum(1 for d in envs_done if d == 0)
        if active_count > 0 and step_criticals > 0:
            critical_per_env = step_criticals / active_count
            for env_idx in range(agent.args.test_env_num):
                if envs_done[env_idx] == 0:
                    current_episode_criticals[env_idx] += critical_per_env

        actions = [agent.list_actions[idx] for idx in action_idxs]
        previous_obs = list(new_obs)
        previous_action = list(actions)

        if len(list_actions) > 6:
            real_a = np.copy(action_idxs)
            real_a[real_a > 6] = 6
            obs, rewards, dones, infos = envs.step(real_a)
        else:
            obs, rewards, dones, infos = envs.step(action_idxs)

        for j in range(n_test_envs):
            work_mem_act[j].append(actions[j])

        for j in range(n_test_envs):
            returns[j] += rewards[j]
            if dones[j] and not envs_done[j]:
                if agent.args.save_gif == 1 and image_raw[j]:
                    _save_episode_gifs(
                        agent, j, image_raw[j], image_gif[j], map_gif[j],
                        agent_path_highlighted_gif[j], agent_positions_all_steps[j],
                        env_missions[j], returns[j],
                    )

                episode_duration = time.time() - episode_start_times[j]
                tokens_per_episode.append(current_episode_tokens[j])
                llm_calls_per_episode.append(current_episode_llm_calls[j])
                episode_durations.append(episode_duration)
                total_tokens_all_episodes += current_episode_tokens[j]
                decisions_per_episode.append(current_episode_decisions[j])
                criticals_per_episode.append(int(current_episode_criticals[j]))

                logger.info(
                    "env %d finished, return: %s, tokens: %d, llm_calls: %d, "
                    "decisions: %d, criticals: %d, time: %.2fs",
                    j, returns[j], current_episode_tokens[j],
                    current_episode_llm_calls[j], current_episode_decisions[j],
                    int(current_episode_criticals[j]), episode_duration,
                )

                envs_done[j] = 1
                test_return_per_episode.append(returns[j])
                obs_queue[j].clear()
                acts_queue[j].clear()
                work_mem_obs[j].clear()
                work_mem_act[j].clear()
                returns[j] = 0
            else:
                acts_queue[j].append(actions[j])
                obs_queue[j].append(infos[j]['descriptions'])
                work_mem[j].append(ec_prompts[j])

        next_prompts = [
            agent.generate_prompt(goal=obs[j]['mission'], list_actions=list_actions, deque_obs=obs_queue[j])
            for j in range(n_test_envs)
        ]
        ec_prompts = next_prompts

        for j in range(n_test_envs):
            workmem_raw = agent._extract_raw_observation(ec_prompts[j])
            work_mem_obs[j].append([workmem_raw])
            new_obs[j] = workmem_raw
            env_missions[j] = obs[j]['mission']

        work_mem_prompts = [
            agent.generate_work_mem(work_mem_obs[j], work_mem_act[j])
            for j in range(n_test_envs)
        ]

        if all(envs_done):
            break

    envs.close()
    logger.info("All test finished, return per episode: %s", test_return_per_episode)

    hit_ratio = peek_counter['non_null'] / peek_counter['total'] if peek_counter['total'] > 0 else 0.0
    logger.info("Test #%d cache hit ratio: %.2f%%", agent.test_times, hit_ratio * 100)

    _log_test_statistics(
        tokens_per_episode, llm_calls_per_episode, episode_durations,
        test_return_per_episode, total_tokens_all_episodes,
        total_decision_count, total_critical_count,
        decisions_per_episode, criticals_per_episode,
    )

    if tokens_per_episode:
        successful = [t for t, r in zip(tokens_per_episode, test_return_per_episode) if r > 0]
        successful_calls = [c for c, r in zip(llm_calls_per_episode, test_return_per_episode) if r > 0]
        successful_times = [d for d, r in zip(episode_durations, test_return_per_episode) if r > 0]
        avg_tokens = sum(successful) / len(successful) if successful else sum(tokens_per_episode) / len(tokens_per_episode)
        avg_calls = sum(successful_calls) / len(successful_calls) if successful_calls else sum(llm_calls_per_episode) / len(llm_calls_per_episode)
        avg_time = sum(successful_times) / len(successful_times) if successful_times else sum(episode_durations) / len(episode_durations)
    else:
        avg_tokens = avg_calls = avg_time = 0

    return (test_return_per_episode, hit_ratio, tokens_per_episode, avg_tokens,
            episode_durations, avg_time, llm_calls_per_episode, avg_calls,
            total_decision_count, total_critical_count,
            decisions_per_episode, criticals_per_episode)


def _save_episode_gifs(agent, j, image_raw, image_gif, map_gif,
                       agent_path_highlighted_gif, agent_positions,
                       mission, ret):
    """Save GIF animations for a completed episode."""
    last_image = image_raw[-2]
    for agent_pos in agent_positions:
        tile_size = 32
        y0, y1 = agent_pos[1] * tile_size, (agent_pos[1] + 1) * tile_size
        x0, x1 = agent_pos[0] * tile_size, (agent_pos[0] + 1) * tile_size
        last_image[y0:y1, x0:x1, :] = [173, 216, 230]
        agent_path_highlighted_gif.append(Image.fromarray(last_image))

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ret_rounded = round(ret, 2)

    if image_gif:
        image_gif[0].save(
            f"{agent.images_save_path[j]}output_animation_{ts}_{mission}_return_{ret_rounded}.gif",
            save_all=True, append_images=image_gif[1:], duration=150, loop=0,
        )
    if map_gif:
        map_gif[0].save(
            f"{agent.map_gif_save_path[j]}output_animation_{ts}_{mission}_return_{ret_rounded}.gif",
            save_all=True, append_images=map_gif[1:-1], duration=200, loop=0,
        )
    if agent_path_highlighted_gif:
        agent_path_highlighted_gif[0].save(
            f"{agent.agent_path_highlighted_save_path[j]}output_animation_{ts}_{mission}_return_{ret_rounded}.gif",
            save_all=True, append_images=agent_path_highlighted_gif[1:], duration=150, loop=0,
        )


def _log_test_statistics(tokens_per_episode, llm_calls_per_episode,
                         episode_durations, test_return_per_episode,
                         total_tokens, total_decisions, total_criticals,
                         decisions_per_episode, criticals_per_episode):
    if not tokens_per_episode:
        logger.warning("No episode statistics collected")
        return

    n_episodes = len(tokens_per_episode)
    avg_tokens = total_tokens / n_episodes
    avg_calls = sum(llm_calls_per_episode) / n_episodes
    avg_time = sum(episode_durations) / n_episodes
    successful_indices = [i for i, r in enumerate(test_return_per_episode) if r > 0]

    logger.info("=== Token, LLM Calls & Time Statistics ===")
    logger.info("Total tokens: %d | Total calls: %d | Total time: %.2fs",
                total_tokens, sum(llm_calls_per_episode), sum(episode_durations))
    logger.info("Episodes: %d | Successful: %d | Failed: %d",
                n_episodes, len(successful_indices), n_episodes - len(successful_indices))
    logger.info("All avg - Tokens: %.2f, Calls: %.2f, Time: %.2fs",
                avg_tokens, avg_calls, avg_time)

    if total_decisions > 0:
        logger.info("Decisions: %d | Critical: %d | Ratio: %.2f%%",
                     total_decisions, total_criticals,
                     total_criticals / total_decisions * 100)

    if successful_indices:
        s_tokens = [tokens_per_episode[i] for i in successful_indices]
        s_calls = [llm_calls_per_episode[i] for i in successful_indices]
        s_times = [episode_durations[i] for i in successful_indices]
        logger.info("Successful avg - Tokens: %.2f, Calls: %.2f, Time: %.2fs",
                     sum(s_tokens) / len(s_tokens),
                     sum(s_calls) / len(s_calls),
                     sum(s_times) / len(s_times))
    else:
        logger.warning("No successful episodes!")
