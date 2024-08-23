import torch
from tqdm import tqdm
from dynamics import dynamics


class MDs:
    def __init__(self, config):
        self.device = config['system']['device']
        self.molecule = config['molecule']['name']
        self.start_state = config['molecule']['start_state']
        self.end_state = config['molecule']['end_state']
        self.num_samples = config['agent']['num_samples']

        self.mds = self._init_mds(config)
        self.target_position = self._init_target_position(config)

    def _init_mds(self, config):
        mds = []
        for _ in tqdm(range(self.num_samples), desc="MDs initialization"):
            md = getattr(dynamics, self.molecule.title())(config, self.start_state)
            mds.append(md)
        return mds

    def _init_target_position(self, config):
        print(f"Get position of {self.end_state} of {self.molecule}")

        target_position = getattr(dynamics, self.molecule.title())(
            config, state=self.end_state
        ).position
        target_position = torch.tensor(
            target_position, dtype=torch.float, device=self.device
        ).unsqueeze(0)

        return target_position

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.num_samples):
            self.mds[i].step(force[i])

    def report(self):
        positions, velocities, forces, potentials = [], [], [], []
        for i in range(self.num_samples):
            position, velocity, force, potential = self.mds[i].report()
            positions.append(position)
            velocities.append(velocity)
            forces.append(force)
            potentials.append(potential)

        positions = torch.tensor(positions, dtype=torch.float, device=self.device)
        velocities = torch.tensor(velocities, dtype=torch.float, device=self.device)
        forces = torch.tensor(forces, dtype=torch.float, device=self.device)
        potentials = torch.tensor(potentials, dtype=torch.float, device=self.device)
        return positions, velocities, forces, potentials

    def reset(self):
        for i in range(self.num_samples):
            self.mds[i].reset()

    def set_temperature(self, temperature):
        for i in range(self.num_samples):
            self.mds[i].set_temperature(temperature)
