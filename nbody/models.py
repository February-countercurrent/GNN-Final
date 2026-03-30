import torch
import torch.nn as nn


class MNN_TwoBody(nn.Module):

    def __init__(self, n_step=50, bs=1, **kwargs):
        super().__init__()
        self.n_step = n_step
        self.bs = bs

        hidden = 128
        input_dim = 4   # (x, y, vx, vy) for a single body
        output_dim = 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.traj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

        self.coeff_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 30),
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch, input_features)
        Returns:
            pred_traj  : (batch, input_features)
            alpha_coeffs: (batch, 30)
        """
        h = self.encoder(x)
        pred_traj = self.traj_head(h)
        alpha_coeffs = self.coeff_head(h)
        return pred_traj, alpha_coeffs
