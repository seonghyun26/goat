import numpy as np
import bgflow as bg


class Flow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _forward(self, *xs, **kwargs):
        raise NotImplementedError()

    def _inverse(self, *xs, **kwargs):
        raise NotImplementedError()

    def forward(self, *xs, inverse=False, **kwargs):
        """
        Forward method of the flow.
        Computes the forward or inverse direction of the flow.

        Parameters
        ----------
        xs : torch.Tensor
            Input of the flow

        inverse : boolean
            Whether to compute the forward or inverse
        """
        if inverse:
            return self._inverse(*xs, **kwargs)
        else:
            return self._forward(*xs, **kwargs)


class CoordinateTransform(Flow):
    def __init__(
        self,
        molecule: str,
        z_matrix: Union[np.ndarray, torch.LongTensor],
        fixed_atoms: np.ndarray,
        normalize_angles: bool = True,
        eps: float = 1e-7,
        enforce_boundaries: bool = True,
        raise_warnings: bool = True,
    ):
        super().__init__()
        self.model = bg.RelativeInternalCoordinateTransformation(
            z_matrix=self.z_matrix,
            fixed_atoms=self.rigid_block,
            normalize_angles=True,
        )
        self._set_dimensions(molecule)
        
    def _set_dimensions(self, molecule):
        if molecule == "alanine":
            self.rigid_block = np.array([6, 8, 9, 10, 14])
            self.z_matrix = np.array([
                [0, 1, 4, 6],
                [1, 4, 6, 8],
                [2, 1, 4, 0],
                [3, 1, 4, 0],
                [4, 6, 8, 14],
                [5, 4, 6, 8],
                [7, 6, 8, 4],
                [11, 10, 8, 6],
                [12, 10, 8, 11],
                [13, 10, 8, 11],
                [15, 14, 8, 16],
                [16, 14, 8, 6],
                [17, 16, 14, 15],
                [18, 16, 14, 8],
                [19, 18, 16, 14],
                [20, 18, 16, 19],
                [21, 18, 16, 19]
            ])
            self.dim_cartesian = len(self.rigid_block) * 3 - 6
            self.dim_bonds = len(self.z_matrix)
            self.dim_angles = dim_bonds
            self.dim_torsions = dim_bonds
        else:
            raise ValueError(f"Unknown molecule: {molecule}")
        
    def forward(self, x, with_pose=True, *args, **kwargs):
        self.model._forward(x, with_pose=with_pose, *args, **kwargs)
        
    def inverse(self, bonds, angles, torsions, x_fixed, **kwargs):
        self.model._inverse(bonds, angles, torsions, x_fixed, **kwargs)

