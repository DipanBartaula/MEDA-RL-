import os
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import CADDataset
from renderer import run_cad_code, render_orthographic
from reward import compute_reward
from model import CADPolicy

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',     type=str, default='./data')
    p.add_argument('--output-dir',   type=str, default='./outputs')
    p.add_argument('--epochs',       type=int, default=10)
    p.add_argument('--batch-size',   type=int, default=1)
    p.add_argument('--lr',           type=float, default=2e-5)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ds = CADDataset(os.path.join(args.data_dir, 'examples.json'))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = CADPolicy(device=device)
    optimizer = AdamW(policy.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        total_reward = 0.0
        policy.model.train()

        for prompts, targets, extents in dl:
            optimizer.zero_grad()
            batch_loss = 0.0

            for prompt, target_img, expected_extents in zip(prompts, targets, extents):
                code = policy.generate(prompt)
                try:
                    mesh = run_cad_code(code)
                    rend = render_orthographic(mesh)
                    reward = compute_reward(rend, target_img, mesh, expected_extents)
                except Exception:
                    reward = 0.0

                logp = policy.get_log_probs(prompt, code)
                loss = - reward * logp
                batch_loss += loss
                total_reward += reward

            batch_loss.backward()
            optimizer.step()

        avg_reward = total_reward / len(ds)
        print(f"Epoch {epoch}/{args.epochs} — avg reward: {avg_reward:.4f}")

        ckpt = os.path.join(args.output_dir, f'model_epoch{epoch}.pt')
        torch.save(policy.model.state_dict(), ckpt)

if __name__ == "__main__":
    main()
