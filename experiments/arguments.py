import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    experiments_dir = Path(__file__).resolve().parent
    project_root = experiments_dir.parent
    agent_dir = experiments_dir / "agents" / "aec"
    logs_dir = project_root / "storage" / "logs"
    models_dir = project_root / "models"
    ec_save_dir = experiments_dir / "ec_save"
    default_spm_path = agent_dir / "spm_models" / "unigram_8k.model"
    default_agent_logs_path = agent_dir / "logs"
    default_sentence_transformer_path = agent_dir / "SentenceEmbeddingsModels" / "bert-sentence-transformers"
    default_llm_path = project_root / "LLM_models" / "Qwen2.5-32B-Instruct"

    # Lamorel-specific arguments
    parser.add_argument("--allow_subgraph_use_with_gradient", action="store_true", default=True, help="Allow subgraph usage with gradient")

    # Distributed setup arguments
    parser.add_argument("--n_rl_processes", type=int, default=1, help="Number of RL processes for distributed setup")
    parser.add_argument("--n_llm_processes", type=int, default=0, help="Number of LLM processes for distributed setup")

    # Accelerate arguments
    parser.add_argument("--config_file", type=str, default="accelerate/default_config.yaml", help="Path to the accelerate configuration file")
    parser.add_argument("--machine_rank", type=int, default=0, help="Machine rank for distributed setup")
    parser.add_argument("--num_machines", type=int, default=None, help="Total number of machines in the distributed setup")
    parser.add_argument("--num_processes", type=int, default=None, help="Total number of processes in the distributed setup")
    parser.add_argument("--main_process_ip", type=str, default=None, help="IP address of the main process")
    parser.add_argument("--main_process_port", type=int, default=12345, help="Port of the main process")

    # LLM-specific arguments
    parser.add_argument("--model_type", type=str, default=None, help="Type of the language model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the language model")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use a pretrained model")
    parser.add_argument("--minibatch_size", type=int, default=None, help="Minibatch size for model training")
    parser.add_argument("--pre_encode_inputs", action="store_true", default=True, help="Pre-encode inputs for LLM processing")

    # Parallelism and GPU usage
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU for LLM processing")
    parser.add_argument("--model_parallelism_size", type=int, default=None, help="Size of model parallelism for distributed LLM training")
    parser.add_argument("--synchronize_gpus_after_scoring", action="store_true", default=False, help="Synchronize GPUs after scoring")
    parser.add_argument("--empty_cuda_cache_after_scoring", action="store_true", default=False, help="Empty CUDA cache after scoring to free up memory")

    # Paths and experiment names
    parser.add_argument("--path", type=str, default=str(project_root), help="Base path for experiment files and logs")
    parser.add_argument("--saving_path_logs", type=str, default=str(logs_dir), help="Path to save training logs")
    parser.add_argument("--saving_path_model", type=str, default=str(models_dir), help="Path to save model checkpoints")
    parser.add_argument("--spm_path", type=str, default=str(default_spm_path), help="Path to the SentencePiece model for LLM tokenization")

    # Experiment settings
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness, ideally set to SLURM_ARRAY_TASK_ID")
    parser.add_argument("--name_experiment", type=str, default="BabyAI", help="Name of the experiment")
    parser.add_argument("--name_model", type=str, default="aec", help="Model name")
    parser.add_argument("--name_environment", type=str, default="goto_gotolocal_pickuplocal", help="Name of the environment")
    # parser.add_argument("--env_name_list", type=list, default=["BabyAI-GoToRedBallNoDists-v0", "BabyAI-GoToLocalS8N7-v0", "BabyAI-PickupLoc-v0"], help="Name of the environment")BabyAI-KeyCorridorS6R3-v0 BabyAI-UnlockLocal-v0  BabyAI-GoToLocal-v0 BabyAI-FindObjS5-v0  BabyAI-PickupLoc-v0 BabyAI-GoToLocal-v0
    parser.add_argument("--env_name_list", type=list, default=["BabyAI-PickupLoc-v0"], help="Name of the environment")
    parser.add_argument("--language", type=str, default="english", help="Language setting for the environment")
    parser.add_argument("--template_test", type=int, default=1, help="Template test type for agent testing")

    # Environment and episode settings 
    parser.add_argument("--number_envs", type=int, default=16, help="Number of parallel environments") 
    parser.add_argument("--num_frames", type=int, default=1500000, help="Total number of steps for training")
    # parser.add_argument("--max_episode_steps", type=int, default=3, help="Maximum steps per episode")
    parser.add_argument("--frames_per_proc", type=int, default=40, help="Frames per process")
    parser.add_argument("--nbr_obs", type=int, default=3, help="Number of observations")

    # Reward and discount settings
    parser.add_argument("--reward_shaping_beta", type=float, default=0, help="Beta value for reward shaping")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor for rewards")

    # Learning and optimization settings
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for the Adam optimizer")
    parser.add_argument("--adam_eps", type=float, default=1e-5, help="Epsilon for Adam optimizer")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="Clipping epsilon for PPO")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training") 
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--gae_lambda", type=float, default=0.99, help="GAE lambda for PPO")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value_loss_coef", type=float, default=0.5, help="Value loss coefficient")

    # Model-specific settings
    parser.add_argument("--action_space", nargs="+", type=str, default=["turn_left", "turn_right", "go_forward", "pick_up", "drop", "toggle"], help="List of actions available in the action space")
    # parser.add_argument("--action_space", nargs="+", type=str, default=["turn_left", "turn_right", "go_forward"], help="List of actions available in the action space")
    parser.add_argument("--load_embedding", action="store_true", default=False, help="Whether to load pretrained embeddings")
    parser.add_argument("--use_action_heads", action="store_true", default=False, help="Whether to use action heads in the model")

    # EMDQN
    parser.add_argument("--emdqn", type=bool, default=True, help="EMDQN otherwise DQN")
    parser.add_argument("--ec_buffer_size", type=int, default=int(5e6), help="episodic memory size")
    parser.add_argument("--ec_latent_dim", type=int, default=384, help="dimensions for random project method")
    parser.add_argument("--gamma", type=float, default=0.99, help="discounted factor")
    parser.add_argument("--save_ec_buffer_path", type=str, default=str(ec_save_dir), help="target network update rate")
    parser.add_argument("--llm_enhance", type=bool, default=True, help="discounted factor")
    parser.add_argument("--qec_weight", type=float, default=0.5, help="discounted factor")
    parser.add_argument("--LLMabs_weight", type=float, default=0.0, help="discounted factor")
    parser.add_argument("--epsilon_greedy_mode", type=str, default='standard', help="epsilon greedy")
    parser.add_argument("--eval_frequency", type=int, default=10, help="eval frequency")
    parser.add_argument("--save_ec_buffer", type=bool, default=True, help="eval frequency")
    parser.add_argument("--test", type=bool, default=False, help="eval frequency")
    parser.add_argument("--model_name", type=str, default="test_work_mem", help="eval frequency")
    parser.add_argument("--start_step", type=int, default=50, help="eval frequency")
    parser.add_argument("--test_env_num", type=int, default=32, help="eval frequency")
    parser.add_argument("--agent_type", type=str, default="aec", help="Agent type")
    parser.add_argument("--agent_logs_path", type=str, default=str(default_agent_logs_path), help="eval frequency")
    parser.add_argument("--save_frequency", type=int, default=100, help="eval frequency")
    parser.add_argument("--sentencetransformer_path", type=str, default=str(default_sentence_transformer_path), help="eval frequency")
    parser.add_argument("--use_api", type=bool, default=False, help="eval frequency")
    parser.add_argument("--llm_name", type=str, default=str(default_llm_path), help="eval frequency")
    parser.add_argument("--update_nums", type=int, default=0, help="eval frequency")
    parser.add_argument("--save_gif", type=int, default=0, help="eval frequency")
    parser.add_argument("--use_necsa", type=bool, default=True, help="eval frequency")
    parser.add_argument("--necsa_weight", type=float, default=0.1, help="eval frequency")
    parser.add_argument("--test_memory_path", type=str, default=None, help="Path to a saved memory directory used for evaluation")
    parser.add_argument("--test_model_path", type=str, default=None, help="Path to a saved model directory used for evaluation")
    
    # DSPy specific arguments
    parser.add_argument("--dspy_lm_model", type=str, default=None, help="Model name for DSPy LM (default to llm_name)")
    parser.add_argument("--dspy_api_base", type=str, default=None, help="API base URL for DSPy LM")
    parser.add_argument("--dspy_api_key", type=str, default=None, help="API key for DSPy LM")
    parser.add_argument("--dspy_optimize_frequency", type=int, default=50, help="Frequency of optimization (in update steps)")
    parser.add_argument("--dspy_trainset_size", type=int, default=200, help="Size of training set for DSPy optimization")
    parser.add_argument("--dspy_optimizer", type=str, default="miprov2", help="Optimizer type: miprov2 or bootstrap")
    parser.add_argument("--dspy_optimized_path", type=str, default=None, help="Path to load/save optimized DSPy module")
    
    # ToT specific arguments
    parser.add_argument("--tot_n_candidates", type=int, default=3, help="Number of candidate thoughts to generate per step in ToT")
    parser.add_argument("--tot_depth", type=int, default=1, help="Depth of thought tree (1 = single-step ToT)")
    
    # SC (Self-Consistency) specific arguments
    parser.add_argument("--sc_n_samples", type=int, default=5, help="Number of CoT samples for Self-Consistency majority voting")
    args = parser.parse_args()

    return args


