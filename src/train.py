import wandb
import torch
import argparse

from tqdm import tqdm
from dynamics.mds import MDs
from dynamics import dynamics
from flow import FlowNetAgent

from utils.logging import Logger
from utils.config import *

args = parse_train_args()

if __name__ == "__main__":
    # Load and set configs
    config = config_init(args)
    torch.manual_seed(config['system']["seed"])
    md = getattr(dynamics, config['molecule']['name'].title())(config, config['molecule']['start_state'])
    logger = Logger(config, md)

    # Initialize Molecular dynamics
    logger.info(f"Initializing {config['training']['num_samples']} MDs starting at {config['molecule']['start_state']}")
    mds = MDs(config)
    agent = FlowNetAgent(config, md, mds)
    temperatures = torch.linspace(
        config['training']['start_temperature'],
        config['training']['end_temperature'],
        config['training']['num_rollouts']
    )
    
    # NOTE: train agent
    logger.info(f"Start training {config['training']['num_rollouts']} rollouts for {config['molecule']['name']}")
    best_loss = float("inf")
    for rollout in range(args.num_rollouts):
        print(f"Rollout: {rollout}")

        log = agent.sample(args, mds, temperatures[rollout])
        logger.log(agent.policy, rollout, **log)

        loss = 0
        for _ in tqdm(range(args.trains_per_rollout), desc="Training"):
            loss += agent.train(args)
        loss = loss / args.trains_per_rollout

        logger.info(f"loss: {loss}")
        if args.wandb:
            wandb.log({"loss": loss}, step=rollout)
        if loss < best_loss:
            best_loss = loss
            torch.save(agent.policy.state_dict(), f"{logger.save_dir}/loss_policy.pt")

    logger.info("Finish training")
    
    # NOTE: Evaluate agent
    config["system"]["type"] = "eval"
    
    
