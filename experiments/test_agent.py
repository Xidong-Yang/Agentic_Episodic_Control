"""
This script run a simple agent in a BabyAI GoTo-Local environment.
"""
# vLLM 在子进程中初始化 CUDA，必须使用 spawn 而非 fork。需在 import 任何库之前设置
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import csv
import logging
import time
from logging import ERROR

import gym
import babyai_text
import babyai.utils as utils

try:
    import wandb
except ImportError as e:
    wandb = None
    logging.warning(f"wandb 导入失败 ({e})，将跳过 wandb 相关功能")

from arguments import parse_args
from runtime import (
    build_envs,
    build_experiment_id,
    create_agent,
    make_reward_function,
    prepare_agent_paths,
    save_reset_images,
)

logger = logging.getLogger(__name__)
gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"] = "offline"

def init_wandb(args, env_name=None, id_expe=None, group=True):
    """Initialize a wandb run with environment-specific naming"""
    if wandb is None:
        return
    run_name = id_expe
    if env_name:
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}_{env_name}"
        
    config = vars(args).copy() if hasattr(args, '__dict__') else args
    
    if group and id_expe:
        wandb.init(
            project="AEC",
            name=run_name,
            config=config,
            tags=[args.name_model, args.name_environment, env_name] if env_name else [args.name_model, args.name_environment],
            group=id_expe,
            reinit=True
        )
    else:
        wandb.init(
            project="AEC",
            name=run_name,
            config=config,
            tags=[args.name_model, args.name_environment, env_name] if env_name else [args.name_model, args.name_environment],
            reinit=True
        )

def run_agent(args, algo, id_expe, agent_logs_path):
    header = (["update", "episodes", "frames", "FPS", "duration"]
              + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["success_rate"]
              + ["reshaped_return_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["reshaped_return_bonus_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
              + ["entropy", "policy_loss", "value_loss", "loss", "grad_norm"]
              + ["success_test_params","success_test_memory_raw","cache_hit_ratio"])
    experiment_path = os.path.join(args.saving_path_logs, id_expe)
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    
    csv_path = os.path.join(experiment_path, 'log.csv')
    # we don't buffer data going in the csv log, because we assume
    # that one update will take much longer than one write to the log
    first_created = not os.path.exists(csv_path)
    csv_writer = csv.writer(open(csv_path, 'a', 1))
    if first_created:
        csv_writer.writerow(header)

    # Restore training status
    status_path = os.path.join(experiment_path, 'status.json')
    # if os.path.exists(status_path):
    #     with open(status_path, 'r') as src:
    #         status = json.load(src)
    # else:
    #     status = {'i': 0,
    #               'num_episodes': 0,
    #               'num_frames': 0}
    format_str = ("\nUpdate: {} | Episodes Done: {} | Frames Seen: {:06} | FPS: {:04.0f} | Ellapsed: {}\
                               \nReward: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Success Rate: {: .2f}\
                               \nReshaped: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f}) | Bonus: {: .2f} +- {: .2f}  (Min: {: .2f} Max: {: .2f})\
                               \nFrames/Eps: {:.1f} +- {:.1f}  (Min: {}, Max {})\
                               \nEntropy: {: .3f} | Policy Loss: {: .3f} | Value Loss: {: .5f} | Loss: {: .3f} | Grad Norm: {: .3f}\
                               \nSuccess Test: {: .2f} | Cache Hit Ratio: {: .2%}")

    total_start_time = time.time()
    for env_num in range(len(args.env_name_list)):
        status = {'i': 0,
        'num_episodes': 0,
        'num_frames': 0}
        args.env_name = args.env_name_list[env_num]
        algo.init_env(env_num)
        
        # 结束之前的wandb运行并为当前环境创建新的wandb运行
        if wandb is not None and wandb.run is not None:
            wandb.finish()
        
        # 初始化当前环境的wandb
        env_name = args.env_name_list[env_num]
        
        
        
        # while status['num_frames'] < args.num_frames: # and not early_stop:
        # algo.number_updates = status['i']
        # logs = algo.update_parameters(env_num)
        
        # Initialize success_test with default values
        success_test = {"mean": 0.0}  # Default value
        hit_ratio = 0.0  # Default cache hit ratio

        # algo.load(args.test_model_path)
        result = algo.test(env_num)
        test_return_per_episode = result[0]
        hit_ratio = result[1]
        success_test = utils.synthesize([1 if r > 0 else 0 for r in test_return_per_episode])
                           
        print(f"🚀 success_test: {success_test['mean']}")
        print(f"🚀 Hit ratio: {hit_ratio}")
        print(f"🚀 test time (min): {(time.time() - total_start_time) / 60}")



# This will be overriden by lamorel's launcher if used
def main(args):
    envs, list_actions = build_envs(args)
    id_expe = build_experiment_id(args)
    agent_logs_path, agent_memory_path = prepare_agent_paths(args, id_expe)
    save_reset_images(envs, agent_logs_path, args.number_envs)
    reshape_reward = make_reward_function(args.reward_shaping_beta)
    algo = create_agent(
        args,
        envs,
        list_actions,
        reshape_reward,
        agent_logs_path,
        agent_memory_path,
    )
    run_agent(args, algo, id_expe, agent_logs_path)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    gym.logger.set_level(ERROR)

    args = parse_args()
    main(args)
