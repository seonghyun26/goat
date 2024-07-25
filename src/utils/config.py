import yaml
import wandb
import argparse


def parse_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config/alanine/c5c7ax.yaml", help="Path to config file")
    parser.add_argument("--date", default="default_date", type=str, help="Date of the training")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    
    # # System Config
    # parser.add_argument("--type", default="train", type=str)
    # parser.add_argument("--molecule", default="alanine", type=str)

    # # Logger Config
    # parser.add_argument("--wandb", action="store_true")
    # parser.add_argument("--project", default="alanine", type=str)
    # parser.add_argument("--save_dir", default="log", type=str)
    # parser.add_argument(
    #     "--save_freq", default=100, type=int, help="Frequency of saving in  rollouts"
    # )

    # # Policy Config
    # parser.add_argument(
    #     "--force", action="store_true", help="Predict force otherwise potential"
    # )

    # # Sampling Config
    # parser.add_argument("--start_state", default="c5", type=str)
    # parser.add_argument("--end_state", default="c7ax", type=str)
    # parser.add_argument("--num_steps", default=1000, type=int, help="Length of paths")
    # parser.add_argument("--feat_aug", default="dist", type=str)
    # parser.add_argument(
    #     "--bias_scale", default=0.01, type=float, help="Scale factor of bias"
    # )
    # parser.add_argument("--scale", default=1, type=float)
    # parser.add_argument("--timestep", default=1, type=float, help="Timestep of integrator")
    # parser.add_argument(
    #     "--sigma", default=0.05, type=float, help="Control reward of arrival"
    # )
    # parser.add_argument(
    #     "--num_samples", default=16, type=int, help="Number of paths to sample"
    # )
    # parser.add_argument(
    #     "--temperature", default=300, type=float, help="Temperature for evaluation"
    # )
    # parser.add_argument("--reward", default="dist", type=str)
    # parser.add_argument("--heavy_atoms", action="store_true")

    # # Training Config
    # parser.add_argument("--start_temperature", default=600, type=float)
    # parser.add_argument("--end_temperature", default=300, type=float)
    # parser.add_argument(
    #     "--max_grad_norm", default=1, type=int, help="Maximum norm of gradient to clip"
    # )
    # parser.add_argument(
    #     "--num_rollouts", default=1000, type=int, help="Number of rollouts (or sampling)"
    # )
    # parser.add_argument(
    #     "--log_z_lr", default=1e-2, type=float, help="Learning rate of estimator for log Z"
    # )
    # parser.add_argument(
    #     "--policy_lr",
    #     default=1e-4,
    #     type=float,
    #     help="Learning rate of bias potential or force",
    # )
    # parser.add_argument(
    #     "--buffer",
    #     default="",
    #     type=str,
    # )
    # parser.add_argument("--prioritized_exp", default=1, type=float)
    # parser.add_argument(
    #     "--buffer_size",
    #     default=2048,
    #     type=int,
    #     help="Size of buffer which stores sampled paths",
    # )
    # parser.add_argument(
    #     "--batch_size", default=16, type=int, help="Batch size for training"
    # )
    # parser.add_argument(
    #     "--trains_per_rollout",
    #     default=2000,
    #     type=int,
    #     help="Number of training per rollout in a rollout",
    # )

    args = parser.parse_args()
    
    return args

def parse_eval_args():
    parser = argparse.ArgumentParser()

    # System Config
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--type", default="eval", type=str)
    parser.add_argument("--best", default="loss", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--molecule", default="alanine", type=str)

    # Logger Config
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--project", default="alanine", type=str)
    parser.add_argument("--save_dir", default="results", type=str)
    parser.add_argument("--date", default="date", type=str, help="Date of the training")

    # Policy Config
    parser.add_argument("--force", action="store_true", help="Network predicts force")

    # Sampling Config
    parser.add_argument("--unbiased", action="store_true")
    parser.add_argument("--start_state", default="c5", type=str)
    parser.add_argument("--end_state", default="c7ax", type=str)
    parser.add_argument("--num_steps", default=1000, type=int, help="Length of paths")
    parser.add_argument("--feat_aug", default="dist", type=str)
    parser.add_argument(
        "--bias_scale", default=0.01, type=float, help="Scale factor of bias"
    )
    parser.add_argument("--scale", default=1, type=float)
    parser.add_argument("--timestep", default=1, type=float, help="Timestep of integrator")
    parser.add_argument(
        "--sigma", default=0.05, type=float, help="Control reward of arrival"
    )
    parser.add_argument(
        "--num_samples", default=64, type=int, help="Number of paths to sample"
    )
    parser.add_argument(
        "--temperature", default=300, type=float, help="Temperature for evaluation"
    )
    parser.add_argument("--reward", default="dist", type=str)
    parser.add_argument("--heavy_atoms", action="store_true")

    args = parser.parse_args()

def config_init(args):
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"No config file {args.config} provided")
    
    args_dict = vars(args)
    config['system'].update(args_dict)
    
    if config['wandb']['use']:
        wandb.init(
            project=config['wandb']['project'],
            config=config
        )
        
    return config
