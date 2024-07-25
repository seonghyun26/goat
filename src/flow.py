import torch
import proxy
import openmm.unit as unit

from tqdm import tqdm
from utils.utils import *
from torch.distributions import Normal

from buffer import ReplayBuffer

class FlowNetAgent:
    def __init__(self, config, md, mds):
        self.a = md.a
        self.num_particles = md.num_particles
        self.std = torch.tensor(
            md.std.value_in_unit(unit.nanometer / unit.femtosecond),
            dtype=torch.float,
            device=config['system']['device'],
        )
        self.m = torch.tensor(
            md.m.value_in_unit(md.m.unit),
            dtype=torch.float,
            device=config['system']['device'],
        ).unsqueeze(-1)
        self.policy = getattr(proxy, config['molecule']['name'].title())(config, md)
        self.heavy_atom_ids = md.heavy_atom_ids
        self.normal = Normal(0, self.std)

        self.scale = config['agent']['scale']
        self.bias_scale = config['agent']['bias_scale']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        position = mds.report()[0]
        self.target_position = kabsch(mds.target_position, position[:1])

        if config['system']['type'] == "train":
            self.replay = ReplayBuffer(config, md)

    def sample(self, config, mds, temperature):
        num_samples = config['agent']['num_samples']
        num_steps = config['agent']['num_steps']
        
        positions = torch.zeros(
            (num_samples, num_steps + 1, self.num_particles, 3),
            device=config['system']['device'],
        )
        velocities = torch.zeros(
            (num_samples, num_steps + 1, self.num_particles, 3),
            device=config['system']['device'],
        )
        actions = torch.zeros(
            (num_samples, num_steps, self.num_particles, 3),
            device=config['system']['device'],
        )
        potentials = torch.zeros(
            (num_samples, num_steps + 1), device=config['system']['device']
        )

        position, velocity, _, potential = mds.report()

        positions[:, 0] = position
        velocities[:, 0] = velocity
        potentials[:, 0] = potential

        mds.set_temperature(temperature)
        for s in tqdm(range(num_steps), desc="Sampling"):
            if config['system']['type'] == "eval" and config['evaluate']['unbiased']:
                bias = torch.zeros(
                    (num_samples, self.num_particles, 3), device=config['system']['device']
                )
            else:
                bias = (
                    self.bias_scale
                    * self.policy(position.detach(), self.target_position)
                    .squeeze()
                    .detach()
                )
            mds.step(bias)

            next_position, velocity, force, potential = mds.report()

            noise = (
                velocity
                + self.a * config['dynamics']['timestep'] * force / self.m
                - self.a * (next_position - position) / config['dynamics']['timestep']
            )

            positions[:, s + 1] = next_position
            potentials[:, s + 1] = potential - (bias * next_position).sum(
                (1, 2)
            )  # Subtract bias potential to get true potential

            position = next_position
            bias = 1e-6 * bias  # kJ/(mol*nm) -> (da*nm)/fs**2
            action = self.a * config['dynamics']['timestep'] * -bias / self.m + noise

            actions[:, s] = action
        mds.reset()

        log_md_reward = self.normal.log_prob(actions).mean((1, 2, 3))

        if config['agent']['reward'] == "kabsch":
            log_target_reward = torch.zeros(
                num_samples, num_steps, device=config['system']['device']
            )
            for i in range(num_samples):
                aligned_target_position = kabsch(self.target_position, positions[i][1:])
                target_velocity = (
                    aligned_target_position - positions[i][:-1]
                ) / config['dynamics']['timestep']
                log_target_reward[i] = -0.5 * torch.square(
                    (target_velocity - velocities[i][1:]) / config['agent']['sigma']
                ).mean((1, 2))
        elif config['agent']['reward'] == "dist":
            target_pd = pairwise_dist(self.target_position)

            log_target_reward = torch.zeros(
                num_samples, num_steps + 1, device=config['system']['device']
            )
            for i in range(num_samples):
                pd = pairwise_dist(positions[i])
                log_target_reward[i] = -0.5 * torch.square(
                    (pd - target_pd) / config['agent']['sigma']
                ).mean((1, 2))
        elif config['agent']['reward'] == "s_dist":
            log_target_reward = torch.zeros(
                num_samples, num_steps + 1, device=config['system']['device']
            )
            for i in range(num_samples):
                if config['agent']['heavy_atoms']:
                    log_target_reward[i] = -(
                        compute_s_dist(
                            positions[i][:, self.heavy_atom_ids],
                            self.target_position[:, self.heavy_atom_ids],
                        )
                        / config['agent']['sigma']
                    )
                else:
                    log_target_reward[i] = -(
                        compute_s_dist(positions[i], self.target_position) / config['agent']['sigma']
                    )

        log_target_reward, last_idx = log_target_reward.max(1)

        log_reward = log_md_reward + log_target_reward

        if config['system']['type'] == "train":
            self.replay.add((positions, actions, log_reward))
        if config['system']['type'] == "eval" and config['evaluate']['unbiased']:
            last_idx = (
                torch.zeros(num_samples, dtype=torch.long, device=config['system']['device'])
                + num_steps
            )

        log = {
            "actions": actions,
            "last_idx": last_idx,
            "positions": positions,
            "potentials": potentials,
            "log_md_reward": log_md_reward,
            "log_target_reward": log_target_reward,
            "target_position": self.target_position,
            "last_position": positions[torch.arange(num_samples), last_idx],
        }
        return log

    def train(self, config):
        optimizer = torch.optim.Adam(
            [
                {"params": [self.policy.log_z], "lr": float(config['training']['log_z_lr'])},
                {"params": self.policy.mlp.parameters(), "lr": float(config['training']['policy_lr'])},
            ]
        )

        indices, positions, actions, log_reward = self.replay.sample()

        biases = self.bias_scale * self.policy(positions, self.target_position)
        biases = 1e-6 * biases[:, :-1]  # kJ/(mol*nm) -> (da*nm)/fs**2
        means = self.a * config['dynamics']['timestep'] * - biases / self.m

        log_z = self.policy.log_z
        log_forward = self.normal.log_prob(actions - means).mean((1, 2, 3))
        tb_error = log_z + log_forward - log_reward
        loss = tb_error.square().mean() * self.scale

        loss.backward()

        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if config["agent"]["buffer"]["prioritized"]:
            self.replay.update_priorities(indices, tb_error.abs().detach())
        return loss.item()

