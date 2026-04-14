from __future__ import annotations

import time
from pathlib import Path

import gym
import numpy as np
from PIL import Image
from babyai.paral_env_simple import ParallelEnv

from agents.aec.aec_agent import AECAgent


EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent


def make_reward_function(reward_shaping_beta):
    def reward_function(subgoal_proba=None, reward=None, policy_value=None, llm_0=None):
        if reward > 0:
            return [20 * reward, 0]
        return [0, 0]

    def reward_function_shaped(subgoal_proba=None, reward=None, policy_value=None, llm_0=None):
        if reward > 0:
            penalty = np.log(subgoal_proba / policy_value)
            return [20 * reward - penalty, -penalty]
        penalty = np.log(subgoal_proba / policy_value)
        return [-penalty, -penalty]

    return reward_function_shaped if reward_shaping_beta != 0 else reward_function


def build_envs(args):
    envs = []
    for env_name in args.env_name_list:
        env_instances = []
        for env_index in range(args.number_envs):
            env = gym.make(env_name)
            env.seed(100 * args.seed + env_index)
            env_instances.append(env)
        envs.append(ParallelEnv(env_instances))
        print(f"Activate environment: {env_name} | Max steps: {env_instances[0].max_steps}")

    list_actions = [action.replace("_", " ") for action in args.action_space]
    return envs, list_actions


def build_experiment_id(args):
    return (
        f"{args.name_experiment}_{args.name_model}_"
        f"seed_{args.seed}_{args.env_name_list[0]}_time{time.strftime('%Y%m%d-%H%M%S')}"
    )


def prepare_agent_paths(args, experiment_id):
    log_root = Path(args.saving_path_logs)
    log_root.mkdir(parents=True, exist_ok=True)
    args.saving_path_logs = str(log_root)
    args.saving_path_model = str(log_root / experiment_id)

    agent_root = Path(args.agent_logs_path)
    agent_root.mkdir(parents=True, exist_ok=True)
    args.agent_logs_path = str(agent_root)

    agent_memory_path = str(agent_root / experiment_id / "memory")
    agent_logs_path = []
    for env_name in args.env_name_list:
        env_log_path = agent_root / experiment_id / env_name
        env_log_path.mkdir(parents=True, exist_ok=True)
        agent_logs_path.append(str(env_log_path))

    return agent_logs_path, agent_memory_path


def save_reset_images(envs, agent_logs_path, number_envs):
    for index, env in enumerate(envs):
        env.reset()
        rendered = env.render(mode="rgb_array")
        images = [item[0] for item in rendered]
        reset_image_path = Path(agent_logs_path[index]) / "env_reset"
        reset_image_path.mkdir(parents=True, exist_ok=True)
        for env_index in range(number_envs):
            image = Image.fromarray(images[env_index])
            image.save(reset_image_path / f"output_image_{env_index}.png")


def create_agent(args, envs, list_actions, reshape_reward, agent_logs_path, agent_memory_path):
    if args.agent_type != "aec":
        raise ValueError(
            f"Unsupported agent_type '{args.agent_type}'. "
            "This open-source version currently supports only 'aec'."
        )

    return AECAgent(
        args,
        envs,
        list_actions,
        reshape_reward,
        args.spm_path,
        saving_agent_logs_path=agent_logs_path,
        saving_agent_memory_path=agent_memory_path,
        max_steps=args.number_envs * 4,
    )
