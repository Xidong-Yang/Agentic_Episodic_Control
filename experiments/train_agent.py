"""
This script run a simple agent in a BabyAI GoTo-Local environment.
"""
import os
# vLLM 通过此环境变量选择 multiprocessing 启动方式，必须为 spawn 才能与 CUDA 兼容
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import csv
import json
import logging
import time
import collections
from logging import ERROR

import numpy as np
import wandb
import gym
import babyai_text
import babyai.utils as utils
from colorama import Fore

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
        if wandb.run is not None:
            wandb.finish()
        
        # 初始化当前环境的wandb
        env_name = args.env_name_list[env_num]
        init_wandb(args, env_name=env_name, id_expe=id_expe, group=True)
        
        # 跟踪最近5次test的成功率
        recent_success_rates = collections.deque(maxlen=5)
        early_stop = False
        
        # Create test metrics CSV file for current environment
        test_metrics_path = os.path.join(agent_logs_path[env_num], 'test_metrics.csv')
        test_metrics_header = ["update", "frames", "success_rate", "cache_hit_ratio"]
        test_metrics_first_created = not os.path.exists(test_metrics_path)
        test_metrics_file = open(test_metrics_path, 'a', 1)
        test_metrics_writer = csv.writer(test_metrics_file)
        if test_metrics_first_created:
            test_metrics_writer.writerow(test_metrics_header)
        
        while status['num_frames'] < args.num_frames: # and not early_stop:

            update_start_time = time.time()
            # algo.number_updates = status['i']
            logs = algo.update_parameters(env_num)
            args.update_nums += 1
            
            # Initialize success_test with default values
            success_test = {"mean": 0.0}  # Default value
            hit_ratio = 0.0  # Default cache hit ratio
            status['num_frames'] += logs["num_frames"]

            if args.test: #and args.update_nums >= args.start_step:
                if args.update_nums % args.eval_frequency == 0:
                    test_return_per_episode, hit_ratio = algo.test(env_num)
                    success_test = utils.synthesize(
                        [1 if r > 0 else 0 for r in test_return_per_episode])
                    
                    # Save test metrics to CSV
                    test_metrics_writer.writerow([
                        args.update_nums, 
                        status['num_frames'], 
                        success_test["mean"],
                        hit_ratio
                    ])
                    
                    # 记录最近的成功率并检查是否满足提前停止条件
                    success_rate = success_test["mean"]
                    recent_success_rates.append(success_rate)
                    
                    # 当收集到5个样本后，检查是否所有成功率都超过93%
                    if len(recent_success_rates) >= 5 and all(rate > 0.85 for rate in recent_success_rates):
                        logger.info(Fore.GREEN + f"Early stopping for environment {env_name}: Last 5 success rates all above 93%" + Fore.RESET)
                        wandb.log({
                            "env": env_name,
                            "early_stopping": True,
                            "early_stopping_reason": "Last 5 success rates all above 93%"
                        })
                        early_stop = True
                    
                    wandb.log({
                        "env": env_name,
                        "success_test": success_test["mean"],
                        "test_return_mean": np.mean(test_return_per_episode),
                        "cache_hit_ratio": hit_ratio,
                    }, commit=False)
                else:
                    success_test = {"mean": 0.0}
            else:
                wandb.log({
                    "env": env_name,
                    "success_test": 0.0,
                    "test_return_mean": 0.0,
                    "cache_hit_ratio": 0.0,
                }, commit=False)

            update_end_time = time.time()
            status['num_episodes'] += logs['episodes_done']
            status['i'] += 1

            total_ellapsed_time = int(time.time() - total_start_time)
            fps = logs["num_frames"] / (update_end_time - update_start_time)

            return_per_episode = utils.synthesize(logs["return_per_episode"])
            success_per_episode = utils.synthesize(
                [1 if r > 0 else 0 for r in logs["return_per_episode"]])
            

            reshaped_return_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            reshaped_return_bonus_per_episode = utils.synthesize(logs["reshaped_return_bonus_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            data = [status['i'], status['num_episodes'], status['num_frames'],
                    fps, total_ellapsed_time,
                    *return_per_episode.values(),
                    success_per_episode['mean'],
                    *reshaped_return_per_episode.values(),
                    *reshaped_return_bonus_per_episode.values(),
                    *num_frames_per_episode.values(),
                    logs["entropy"], logs["policy_loss"], logs["value_loss"],
                    logs["loss"], logs["grad_norm"],success_test["mean"],hit_ratio]

            logger.info(Fore.YELLOW + format_str.format(*data) + Fore.RESET)
            csv_writer.writerow(data)

            # Log to wandb with environment info
            wandb.log({
                "update": args.update_nums,
                "num_episodes": status['num_episodes'],
                "num_frames": status['num_frames'],
                "fps": fps,
                "duration": total_ellapsed_time,
                "return_mean": return_per_episode["mean"],
                "return_std": return_per_episode["std"],
                "return_min": return_per_episode["min"],
                "return_max": return_per_episode["max"],
                "success_rate": success_per_episode["mean"],
                "reshaped_return_mean": reshaped_return_per_episode["mean"],
                "reshaped_return_std": reshaped_return_per_episode["std"],
                "reshaped_return_min": reshaped_return_per_episode["min"],
                "reshaped_return_max": reshaped_return_per_episode["max"],
                "reshaped_return_bonus_mean": reshaped_return_bonus_per_episode["mean"],
                "reshaped_return_bonus_std": reshaped_return_bonus_per_episode["std"],
                "num_frames_mean": num_frames_per_episode["mean"],
                "entropy": logs["entropy"],
                "policy_loss": logs["policy_loss"],
                "value_loss": logs["value_loss"],
                "loss": logs["loss"],
                # "success_test": success_test["mean"]
                # "cache_hit_ratio": hit_ratio
            })

            with open(status_path, 'w') as dst:
                json.dump(status, dst)
        
        # 完成当前环境的训练后结束wandb运行
        wandb.finish()
        
        # Close the test metrics file
        test_metrics_file.close()


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
