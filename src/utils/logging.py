import os
import sys
import wandb
import torch
import logging

from .plot import *
from .metrics import Metric

def create_folders(path, folders):
    for folder in folders:
        path_to_check = f"{path}/{folder}"
        if not os.path.exists(path_to_check):
            os.makedirs(path_to_check)

class Logger:
    def __init__(self, config, md):
        self.type = "train"
        self.wandb = config['wandb']['use']
        self.molecule = config['molecule']['name']
        self.start_file = md.start_file
        self.heavy_atoms = config['agent']['heavy_atoms']
        
        self.metric = Metric(config, md)
        self.best_epd = float("inf")
        self.best_elr = float("-inf")
        
        self.save_freq = config['logger']['save_freq'] if self.type == "train" else 100
        self.save_dir = os.path.join(
            config['logger']['save_dir'],
            config['wandb']['project'],
            config['system']['date'],
            str(config['system']['seed']),
            self.type
        )
        create_folders(f"{self.save_dir}", [
            "paths",
            "path",
            "potentials",
            "potential",
            "etps",
            "efps",
            "policies",
            "3D_views",
        ])

        # Logger basic configurations
        self.logger = logging.getLogger("tps-goal")
        self.logger.setLevel(logging.INFO)

        # File handler
        log_file = self.type + ".log"
        log_file = os.path.join(self.save_dir, log_file)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        self.logger.info(f"<== Configurations ==>")
        for k, v in config.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info(f"<====================>\n")

    def info(self, message):
        self.logger.info(message)

    def set_eval_mode(self, config):
        self.type = "eval"
        self.save_dir = os.path.join(
            config["logger"]["save_dir"],
            config["wandb"]["project"],
            config["system"]["date"],
            str(config["system"]["seed"]),
            self.type,
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        create_folders(self.save_dir, [
            "potentials",
            "traj",
            "2D_plots"
        ])
    
    def log(
        self,
        policy,
        rollout,
        actions,
        last_idx,
        positions,
        potentials,
        last_position,
        target_position,
        log_md_reward,
        log_target_reward,
    ):
        # Calculate metrics
        if self.molecule in ["alanine", "histidine", "chignolin"]:
            thp, etps, etp_idxs, etp, std_etp, efps, efp_idxs, efp, std_efp = (
                self.metric.cv_metrics(
                    last_idx, last_position, target_position, potentials
                )
            )
        if self.molecule == "chignolin":
            asp3od_thr6og, asp3n_thr8o = chignolin_h_bond(positions)
            eat36, std_at36 = asp3od_thr6og.mean().item(), asp3od_thr6og.std().item()
            eat38, std_at38 = asp3n_thr8o.mean().item(), asp3n_thr8o.std().item()

        ermsd, std_rmsd = self.metric.rmsd(last_position, target_position)
        ll, std_ll = self.metric.log_likelihood(actions)
        epd, std_pd = self.metric.pairwise_distance(last_position, target_position)
        elpd, std_lpd = self.metric.log_pairwise_distance(
            last_position, target_position, self.heavy_atoms
        )
        epcd, std_pcd = self.metric.pairwise_coulomb_distance(
            last_position, target_position
        )
        len, std_len = last_idx.float().mean().item(), last_idx.float().std().item()

        elmr, std_lmr = log_md_reward.mean().item(), log_md_reward.std().item()
        eltr, std_ltr = log_target_reward.mean().item(), log_target_reward.std().item()
        log_reward = log_md_reward + log_target_reward
        elr, std_lr = log_reward.mean().item(), log_reward.std().item()

        # Create dictionary for logging values
        log = {
            "log_z": policy.log_z.item(),
            "ll": ll,
            "epd": epd,
            "elpd": elpd,
            "epcd": epcd,
            "elr": elr,
            "elmr": elmr,
            "eltr": eltr,
            "ermsd": ermsd,
            "len": len,
            "std_ll": std_ll,
            "std_pd": std_pd,
            "std_lpd": std_lpd,
            "std_pcd": std_pcd,
            "std_lr": std_lr,
            "std_lmr": std_lmr,
            "std_ltr": std_ltr,
            "std_rmsd": std_rmsd,
            "std_len": std_len,
        }
        if self.molecule in ["alanine", "histidine", "chignolin"]:
            log.update({
                "thp": thp,
                "etp": etp,
                "efp": efp,
                "std_etp": std_etp,
                "std_efp": std_efp,
            })
        elif self.molecule == "chignolin":
            log.update({
                "eat36": eat36,
                "eat38": eat38,
                "std_at36": std_at36,
                "std_at38": std_at38,
            })
        
        # Training
        if self.type == "train":
            # Print metrics in console
            self.logger.info("-----------------------------------------------------------")
            self.logger.info(f"Rollout: {rollout}")
            self.logger.info(log)
            self.logger.info("-----------------------------------------------------------")
            
            # Save best policies and value
            if epd < self.best_epd:
                self.best_epd = epd
                torch.save(policy.state_dict(), f"{self.save_dir}/epd_policy.pt")
            if elr > self.best_elr:
                self.best_elr = elr
                torch.save(policy.state_dict(), f"{self.save_dir}/elr_policy.pt")
                
            # Save policy and potential plot at each frequency
            if rollout % self.save_freq == 0:
                torch.save(policy.state_dict(), f"{self.save_dir}/policies/{rollout}.pt")
                fig_pot = plot_potentials(self.save_dir, rollout, potentials, last_idx)
                
                if self.molecule == "alanine":
                    fig_path = plot_paths_alanine(
                        self.save_dir, rollout, positions, target_position, last_idx
                    )
                elif self.molecule == "chignolin":
                    fig_path = plot_paths_chignolin(
                        self.save_dir, rollout, positions, last_idx
                    )
        
        # Evaluation
        elif self.type == "eval":
            if self.molecule == "alanine":
                plot_path_alanine(self.save_dir, positions, target_position, last_idx)
            elif self.molecule == "chignolin":
                plot_path_chignolin(self.save_dir, positions, last_idx)
            
            plot_potential(self.save_dir, potentials, last_idx)
            plot_3D_view(
                self.save_dir, self.start_file, positions, potentials, last_idx
            )
            for old_key in log.keys():
                log["eval/" + str(old_key)] = log[old_key]
                del log[old_key]
        
        else:
            raise ValueError(f"Invalid type {self.type}")    
        
        # Log to wandb
        if self.wandb:
            log.update({"potentials": wandb.Image(fig_pot)})
            if self.molecule in ["alanine", "histidine", "chignolin"]:
                fig_etp = plot_etps(self.save_dir, rollout, etps, etp_idxs)
                fig_efp = plot_efps(self.save_dir, rollout, efps, efp_idxs)
                log.update({
                    "paths": wandb.Image(fig_path),
                    "etps": wandb.Image(fig_etp),
                    "efps": wandb.Image(fig_efp),
                })
            if self.type == "train":
                wandb.log(log, step=rollout)
            elif self.type == "eval":
                wandb.log(log)

        

