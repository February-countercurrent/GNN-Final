import sys
import os
import torch
import torch.nn as nn

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

try:
    from nbody.models import MNN_TwoBody as TeamBaseMNN
    print("Successfully imported Two-Body MNN model")
except ImportError:
    print("Warning: Could not automatically resolve 'nbody.models'. Please verify the exact filename inside the 'nbody/' folder.")
    
    class TeamBaseMNN(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.dummy_param = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            # to mock the (predicted_trajectory, alpha_coefficients) output
            return x, torch.zeros((x.shape[0], 30), device=x.device)

class BaseMNN(nn.Module):
    def __init__(self, **kwargs):
        super(BaseMNN, self).__init__()
        # Instantiate the team's model
        self.model = TeamBaseMNN(**kwargs)

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, tuple):
            pred_traj, alpha_coeffs = output[0], output[1]
        else:
            pred_traj = output
            alpha_coeffs = torch.zeros((x.shape[0], 30), device=x.device, requires_grad=True)
            
        return pred_traj, alpha_coeffs