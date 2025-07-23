import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_reward(rendered: np.ndarray,
                   target: np.ndarray,
                   mesh,
                   expected_extents: tuple,
                   alpha: float = 0.3,
                   beta:  float = 0.5,
                   gamma: float = 0.2
                  ) -> float:
    """
    Composite reward ∈ [0,1] = α·MSE + β·SSIM + γ·DimensionAccuracy.

    - rendered, target: H×W floats [0,1]
    - mesh: trimesh.Trimesh
    - expected_extents: (dx,dy,dz)
    """
    # 1) MSE term
    mse = np.mean((np.clip(rendered,0,1) - np.clip(target,0,1))**2)
    mse_reward = max(0.0, 1.0 - mse)

    # 2) SSIM term ∈ [0,1]
    ssim_raw = ssim(target, rendered, data_range=1.0)
    ssim_reward = (ssim_raw + 1.0) / 2.0

    # 3) Dimension term
    actual = mesh.extents  # array([dx,dy,dz])
    expected = np.array(expected_extents, dtype=float)
    eps = 1e-6
    rel_err = np.abs(actual - expected) / (expected + eps)
    dim_reward = float(np.clip(1.0 - np.mean(rel_err), 0.0, 1.0))

    # Weighted sum
    reward = alpha * mse_reward + beta * ssim_reward + gamma * dim_reward
    return float(np.clip(reward, 0.0, 1.0))
