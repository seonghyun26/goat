import numpy as np
import openmm.unit as unit
from abc import abstractmethod, ABC
from scipy.constants import physical_constants

nuclear_charge = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
}


class BaseDynamics(ABC):
    def __init__(self, config, state):
        super().__init__()
        self.start_file = f"./data/{config['molecule']['name']}/{state}.pdb"

        self.temperature = config['dynamics']['temperature'] * unit.kelvin
        self.friction_coefficient = 1 / unit.picoseconds
        self.timestep = config['dynamics']['timestep'] * unit.femtoseconds

        self.pdb, self.integrator, self.simulation, self.external_force = self.setup()

        self.simulation.minimizeEnergy()
        self.position = self.report()[0]
        self.reset()

        self.num_particles = self.simulation.system.getNumParticles()

        self.a, self.m, self.std = self.get_md_info()
        self.charge_matrix = self.get_charge_matrix()
        self.heavy_atom_ids = self.get_heavy_atoms()

    @abstractmethod
    def setup(self):
        pass

    def get_md_info(self):
        a = np.exp(-self.timestep * self.friction_coefficient)

        m = [
            self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(self.num_particles)
        ]
        m = unit.Quantity(np.array(m), unit.dalton)

        unadjusted_variance = (
            unit.BOLTZMANN_CONSTANT_kB * self.temperature * (1 - a**2) / m[:, None]
        )
        std_SI_units = (
            1
            / physical_constants["unified atomic mass unit"][0]
            * unadjusted_variance.value_in_unit(unit.joule / unit.dalton)
        )
        std = unit.Quantity(np.sqrt(std_SI_units), unit.meter / unit.second)
        return a, m, std

    def get_charge_matrix(self):
        charge_matrix = np.zeros((self.num_particles, self.num_particles))
        topology = self.pdb.getTopology()
        for i, atom_i in enumerate(topology.atoms()):
            for j, atom_j in enumerate(topology.atoms()):
                if i == j:
                    charge_matrix[i, j] = 0.5 * nuclear_charge[
                        atom_i.element.symbol
                    ] ** (2.4)
                else:
                    charge_matrix[i, j] = (
                        nuclear_charge[atom_i.element.symbol]
                        * nuclear_charge[atom_j.element.symbol]
                    )
        return charge_matrix

    def get_heavy_atoms(self):
        heavy_atom_ids = []
        topology = self.pdb.getTopology()
        for atom in topology.atoms():
            if atom.element.symbol != "H":
                heavy_atom_ids.append(atom.index)
        return heavy_atom_ids

    def step(self, forces):
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True, getEnergy=True
        )
        positions = state.getPositions().value_in_unit(unit.nanometer)
        velocities = state.getVelocities().value_in_unit(
            unit.nanometer / unit.femtosecond
        )
        forces = state.getForces().value_in_unit(
            unit.dalton * unit.nanometer / unit.femtosecond / unit.femtosecond
        )
        potentials = state.getPotentialEnergy().value_in_unit(
            unit.kilojoules / unit.mole
        )
        return positions, velocities, forces, potentials

    def reset(self):
        for i in range(len(self.position)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(0)

    def set_temperature(self, temperature):
        self.integrator.setTemperature(temperature * unit.kelvin)
