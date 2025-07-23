# CAD‑RL‑Trainer (with LLaMA‑7B & Dimension‐Aware Reward)

Train a LLaMA‑2 7B policy to generate CAD query code via GRPO, with a **composite reward**:
- **Pixel MSE** (1−normalized‑MSE)
- **SSIM** (structural similarity)
- **Dimension accuracy** (mesh extents vs. expected)

## Setup

```bash
git clone <repo_url> cad-rl-trainer
cd cad-rl-trainer
./run.sh
