import copy
import json
import logging
import os
import shutil
from collections import deque

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

from agents.base_agent import BaseAgent
from .constants import ENV_DESCRIPTIONS, WORK_MEM_OBS_MAXLEN, WORK_MEM_ACT_MAXLEN
from .llm_backend import LLMBackend
from .utils.lru_knn import LRUKnn
from .utils.memory import State
from .observation import llm_abstract_obs
from .world_model import update_world_models
from .action_selection import select_actions
from .testing import run_test

logger = logging.getLogger(__name__)


class AECAgent(BaseAgent):
    def __init__(self, args, envs, list_actions, reshape_reward, spm_path,
                 saving_agent_logs_path, saving_agent_memory_path,
                 gamma=0.9, max_steps=64):
        super().__init__(envs)
        self.args = args
        self.list_actions = list_actions
        self.reshape_reward = reshape_reward
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_path)
        self.saving_agent_logs_path = saving_agent_logs_path
        self.saving_agent_memory_path = saving_agent_memory_path
        self.gamma = gamma
        self.max_steps = max_steps
        self.action_size = len(list_actions)
        self.flag_update_kdtree = [False] * self.action_size

        if not hasattr(self.args, 'critical_variance_threshold'):
            self.args.critical_variance_threshold = 5

        self.ec_save_path = self.saving_agent_memory_path + '/buffer_text'

        self.ec_buffer_LLMabs = []
        for j in range(self.action_size):
            self.ec_buffer_LLMabs.append(LRUKnn(
                self.args, self.saving_agent_memory_path, self.action_size,
                self.args.ec_buffer_size, self.args.ec_latent_dim,
                env_name=self.args.name_environment,
                save_qa_path=self.ec_save_path, save_qa_index=j,
                save_type='LLMabs',
            ))

        self.llm_backend = LLMBackend(self.args.llm_name)

    # ------------------------------------------------------------------ #
    #  Prompt helpers
    # ------------------------------------------------------------------ #

    def _split_prompt_sections(self, prompt):
        parts = prompt.split('LLM abstract:')
        raw_section = parts[0].strip()
        llm_abstract = parts[1].strip() if len(parts) > 1 else ''
        return raw_section, llm_abstract

    def _extract_raw_observation(self, prompt):
        raw_section, _ = self._split_prompt_sections(prompt)
        raw_parts = raw_section.split('Observation : ')
        return raw_parts[1] if len(raw_parts) > 1 else raw_parts[0]

    def _extract_llm_abstract_section(self, prompt):
        _, llm_abstract = self._split_prompt_sections(prompt)
        return f'LLM abstract:{llm_abstract}' if llm_abstract else prompt

    def generate_work_mem(self, deque_work_mem_obs, deque_work_mem_act):
        ldwmo = len(deque_work_mem_obs)
        ldwma = len(deque_work_mem_act)
        work_mem = "\n **Note: T-1 refers to the previous step, T-2 refers to two steps ago, and so on.**"

        for i in range(ldwmo):
            is_current = (i == ldwmo - 1)
            label = "Current Observation: " if is_current else f"T-{ldwmo - i - 1} - Observation: "
            work_mem += f"\n {label}"
            for d_obs in deque_work_mem_obs[i]:
                work_mem += "{}, ".format(d_obs)
            if i < ldwma and not is_current:
                work_mem += f"\n T-{ldwmo - i - 1} - Action: {deque_work_mem_act[i]}"

        return work_mem

    def build_state(self, obs):
        return [State(self.sp.EncodeAsIds(o)) for o in obs]

    def encode_actions(self, acts):
        return [self.sp.EncodeAsIds(a) for a in acts]

    # ------------------------------------------------------------------ #
    #  Action selection (delegates to action_selection module)
    # ------------------------------------------------------------------ #

    def act(self, states_prompts, poss_acts, step, model='raw',
            max_epsilon=1.0, min_epsilon=0.1, decay_steps=500):
        return select_actions(
            self, states_prompts, poss_acts, step, model,
            max_epsilon, min_epsilon, decay_steps,
        )

    # ------------------------------------------------------------------ #
    #  Environment initialization
    # ------------------------------------------------------------------ #

    def init_env(self, env_num):
        self.obs, self.infos = self.env[env_num].reset()
        self.env_description = ENV_DESCRIPTIONS.get(
            self.args.env_name_list[env_num], ''
        )

        self.n_envs = len(self.obs)
        self.obs_queue = [deque([], maxlen=1) for _ in range(self.n_envs)]
        for j in range(self.n_envs):
            self.obs_queue[j].append(self.infos[j]['descriptions'])

        self.ec_prompts = ''
        prompts = [
            self.generate_prompt(
                goal=self.obs[j]['mission'],
                list_actions=self.list_actions,
                deque_obs=self.obs_queue[j],
            )
            for j in range(self.n_envs)
        ]

        self.encoded_actions = self.encode_actions(self.list_actions)

        self.logs = {
            "return_per_episode": [],
            "reshaped_return_per_episode": [],
            "reshaped_return_bonus_per_episode": [],
            "num_frames_per_episode": [],
            "num_frames": self.max_steps,
            "episodes_done": 0.0,
            "entropy": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "grad_norm": 0.0,
            "loss": 0.0,
        }

        self.llm_prompt_output_path = self.saving_agent_logs_path[env_num] + "/llm_prompt_output"
        if os.path.exists(self.llm_prompt_output_path):
            shutil.rmtree(self.llm_prompt_output_path)
        os.makedirs(self.llm_prompt_output_path, exist_ok=True)

        self.returns = [0] * self.n_envs
        self.reshaped_returns = [0] * self.n_envs
        self.frames_per_episode = [0] * self.n_envs
        self.inner_counter = 0
        self.test_times = 0

        self.episode_data_prompt = [[] for _ in range(self.n_envs)]
        if 'llmabs' in self.args.model_name.lower():
            self.ec_prompts, _, _ = llm_abstract_obs(
                self.llm_backend, prompts, self.env_description,
            )
        else:
            self.ec_prompts = prompts

        self.work_mem_obs = [deque([], maxlen=WORK_MEM_OBS_MAXLEN) for _ in range(self.n_envs)]
        self.work_mem_act = [deque([], maxlen=WORK_MEM_ACT_MAXLEN) for _ in range(self.n_envs)]
        self.world_models = [None] * self.n_envs
        self.previous_obs = [None] * self.n_envs
        self.previous_action = [None] * self.n_envs
        self.new_obs = [None] * self.n_envs
        self.env_mission = [None] * self.n_envs

        for j in range(self.n_envs):
            workmem_raw = self._extract_raw_observation(self.ec_prompts[j])
            self.new_obs[j] = workmem_raw
            self.work_mem_obs[j].append([workmem_raw])
            self.env_mission[j] = self.obs[j]['mission']

        self.work_mem_prompts = [
            self.generate_work_mem(self.work_mem_obs[j], self.work_mem_act[j])
            for j in range(self.n_envs)
        ]
        self.world_models, _, _ = update_world_models(
            self.llm_backend,
            self.world_models, self.previous_obs, self.previous_action, self.new_obs,
        )

        self._init_test_save_paths(env_num)

    def _init_test_save_paths(self, env_num):
        self.images_save_path = []
        self.map_gif_save_path = []
        self.agent_path_highlighted_save_path = []
        base = self.saving_agent_logs_path[env_num]
        env_name = self.args.name_environment
        seed = self.args.seed
        for j in range(self.args.test_env_num):
            paths = [
                f"{base}/test_gif_{env_name}_{seed}/agent_path/agent_path_env_{j}/",
                f"{base}/test_gif_{env_name}_{seed}/agent_explore/agent_explore_env_{j}/",
                f"{base}/test_gif_{env_name}_{seed}/agent_path_highlighted/agent_path_highlighted_env_{j}/",
            ]
            for p in paths:
                if os.path.exists(p):
                    shutil.rmtree(p)
                os.makedirs(p, exist_ok=True)
            self.images_save_path.append(paths[0])
            self.map_gif_save_path.append(paths[1])
            self.agent_path_highlighted_save_path.append(paths[2])

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #

    def update_parameters(self, env_num):
        episodes_done = 0
        for i in tqdm(range(self.max_steps // self.n_envs), ascii=" " * 9 + ">", ncols=100):
            action_ids, action_idxs, _ = self.act(
                self.ec_prompts, self.encoded_actions,
                step=self.inner_counter, model=self.args.model_name,
            )
            actions = [self.list_actions[idx] for idx in action_idxs]
            self.previous_action = copy.deepcopy(actions)
            self.previous_obs = copy.deepcopy(self.new_obs)

            if len(self.list_actions) > 6:
                real_a = np.copy(action_idxs)
                real_a[real_a > 6] = 6
                obs, rewards, dones, infos = self.env[env_num].step(real_a)
            else:
                obs, rewards, dones, infos = self.env[env_num].step(action_idxs)

            reshaped_rewards = [self.reshape_reward(reward=r)[0] for r in rewards]

            for j in range(self.n_envs):
                self.work_mem_act[j].append(actions[j])

            for j in range(self.n_envs):
                self.returns[j] += rewards[j]
                self.reshaped_returns[j] += reshaped_rewards[j]
                self.frames_per_episode[j] += 1
                self.obs_queue[j].append(infos[j]['descriptions'])
                if dones[j]:
                    logger.info("Episode done, return: %s", self.returns[j])
                    episodes_done += 1
                    self.logs["num_frames_per_episode"].append(self.frames_per_episode[j])
                    self.frames_per_episode[j] = 0
                    self.logs["return_per_episode"].append(self.returns[j])
                    self.returns[j] = 0
                    self.logs["reshaped_return_per_episode"].append(self.reshaped_returns[j])
                    self.logs["reshaped_return_bonus_per_episode"].append(self.reshaped_returns[j])
                    self.reshaped_returns[j] = 0
                    self.work_mem_obs[j].clear()
                    self.work_mem_act[j].clear()
                    self.world_models[j] = None
                    self.previous_obs[j] = None
                    self.previous_action[j] = None

            next_prompts = [
                self.generate_prompt(
                    goal=obs[j]['mission'],
                    list_actions=self.list_actions,
                    deque_obs=self.obs_queue[j],
                )
                for j in range(self.n_envs)
            ]

            for j in range(self.n_envs):
                self.episode_data_prompt[j].append({
                    "env": j,
                    "observation": self.ec_prompts[j],
                    "action": actions[j],
                    "reward": reshaped_rewards[j],
                })
                if dones[j]:
                    self.update_ec(self.episode_data_prompt[j])
                    self.episode_data_prompt[j].clear()

            if 'llmabs' in self.args.model_name.lower():
                self.ec_prompts, _, _ = llm_abstract_obs(
                    self.llm_backend, next_prompts, self.env_description,
                )
            else:
                self.ec_prompts = next_prompts

            for j in range(self.n_envs):
                workmem_raw = self._extract_raw_observation(self.ec_prompts[j])
                self.work_mem_obs[j].append([workmem_raw])
                self.new_obs[j] = workmem_raw
                self.env_mission[j] = obs[j]['mission']

            self.work_mem_prompts = [
                self.generate_work_mem(self.work_mem_obs[j], self.work_mem_act[j])
                for j in range(self.n_envs)
            ]
            self.world_models, _, _ = update_world_models(
                self.llm_backend,
                self.world_models, self.previous_obs, self.previous_action, self.new_obs,
            )

        self.inner_counter += 1
        logger.info("inner_counter: %d", self.inner_counter)

        if self.inner_counter >= self.args.start_step and self.flag_update_kdtree:
            if self.inner_counter % 10 == 0 and self.args.emdqn:
                self.update_kdtree()

        logs = {}
        for k, v in self.logs.items():
            logs[k] = v[:-episodes_done] if isinstance(v, list) else v
        logs["episodes_done"] = episodes_done
        return logs

    # ------------------------------------------------------------------ #
    #  Episodic memory
    # ------------------------------------------------------------------ #

    def update_ec(self, sequence_prompt):
        mt_rew = 0.0
        for seq_prompt in reversed(sequence_prompt):
            env_num = seq_prompt['env']
            o_prompt = seq_prompt['observation']
            u_prompt = seq_prompt['action']
            r_prompt = seq_prompt['reward']
            _, llm_abstract = self._split_prompt_sections(o_prompt)
            obs_LLMabs = f'LLM abstract: {llm_abstract}' if llm_abstract else ''
            index = self.list_actions.index(u_prompt)
            embedding = self.sentece_transformer_model.encode(
                obs_LLMabs, show_progress_bar=False,
            )
            mt_rew = r_prompt + self.args.gamma * mt_rew
            qd, _, _, _ = self.ec_buffer_LLMabs[index].peek(
                embedding, mt_rew, modify=True, save_flag=True,
            )
            self.ec_buffer_LLMabs[index].save(index)
            self.flag_update_kdtree[index] = True
            if qd is None:
                self.ec_buffer_LLMabs[index].add(embedding, mt_rew, obs_LLMabs, env_num)

    def update_kdtree(self):
        for u in range(self.action_size):
            if self.flag_update_kdtree[u]:
                self.ec_buffer_LLMabs[u].update_kdtree()

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save_llm_interaction(self, file_path, data_to_save):
        """Append LLM interaction data to a JSON file."""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(data_to_save)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------ #
    #  Testing (delegates to testing module)
    # ------------------------------------------------------------------ #

    def test(self, env_num):
        return run_test(self, env_num)
