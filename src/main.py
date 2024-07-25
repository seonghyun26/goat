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
    # Load and set config
    config = config_init(args)
    torch.manual_seed(config['system']["seed"])
    md = getattr(dynamics, config['molecule']['name'].title())(config, config['molecule']['start_state'])
    logger = Logger(config, md)

    # Initialize Molecular dynamics
    logger.info(f"Initializing {config['agent']['num_samples']} MDs starting at {config['molecule']['start_state']}")
    mds = MDs(config)
    agent = FlowNetAgent(config, md, mds)
    temperatures = torch.linspace(
        config['training']['start_temperature'],
        config['training']['end_temperature'],
        config['training']['num_rollouts']
    )
    logger.info(f"Done..!\n")
    
    # NOTE: train agent
    logger.info(f"Training {config['training']['num_rollouts']} rollouts for {config['molecule']['name']}")
    best_loss = float("inf")
    for rollout in range(config['training']['num_rollouts']):
        print(f"Rollout: {rollout}")

        log = agent.sample(config, mds, temperatures[rollout])
        logger.log(agent.policy, rollout, **log)

        loss = 0
        for _ in tqdm(range(config['training']['trains_per_rollout']), desc="Training"):
            loss += agent.train(config)
        loss = loss / config['training']['trains_per_rollout']

        logger.info(f"loss: {loss}")
        if config['wandb']['use']:
            wandb.log({"loss": loss}, step=rollout)
        if loss < best_loss:
            best_loss = loss
            torch.save(agent.policy.state_dict(), f"{logger.save_dir}/loss_policy.pt")

    logger.info("Finished training...!\n")
    
    # NOTE: Evaluate agent
    logger.info(f"Evaluation...")
    config = update_config_eval(config)
    mds = MDs(config)
    agent = FlowNetAgent(config, md, mds)
    logger.set_eval_mode(config)
    
    if not config["evaluate"]["unbiased"]:
        model_path = (os.path.join(
            logger.save_dir,
            f"{args.best}_policy.pt",
        ))
        agent.policy.load_state_dict(torch.load(model_path))

    log = agent.sample(config, mds, config["dynamics"]["temperature"])
    logger.log(agent.policy, 0, **log)
    
    logger.info("Finished evaluation..!\n")