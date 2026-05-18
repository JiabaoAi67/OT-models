"""Train all OT methods on the 2D checkerboard.

Usage:
    python train.py --method flow_matching
    python train.py --method si_ode
    python train.py --method si_sde
    python train.py --method dsbm
    python train.py --method nfdm
"""

import argparse
import os
import time

import torch

from ot.data import checkerboard
from ot.models import VelocityMLP, ScoreMLP
from ot.flow_matching import flow_matching_loss, flow_matching_sample
from ot.stochastic_interpolant import (
    si_velocity_loss,
    si_score_loss,
    si_ode_sample,
    si_sde_sample,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="flow_matching",
        choices=["flow_matching", "si_ode", "si_sde", "dsbm", "nfdm"],
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Method: {args.method} | Device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.manual_seed(42)

    # ========================================================================
    # Flow Matching
    # ========================================================================
    if args.method == "flow_matching":
        model = VelocityMLP(args.hidden_dim).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        t0 = time.time()
        for i in range(args.iterations):
            optim.zero_grad()
            x_1 = checkerboard(args.batch_size, device)
            x_0 = torch.randn_like(x_1)
            loss = flow_matching_loss(model, x_0, x_1)
            loss.backward()
            optim.step()

            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  iter {i+1:6d} | "
                    f"{elapsed*1000/args.print_every:5.1f} ms/step | "
                    f"loss {loss.item():.4f}"
                )
                t0 = time.time()

        ckpt_path = os.path.join(args.ckpt_dir, "flow_matching.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

    # ========================================================================
    # Stochastic Interpolants (ODE or SDE)
    # ========================================================================
    elif args.method in ("si_ode", "si_sde"):
        velocity_model = VelocityMLP(args.hidden_dim).to(device)
        score_model = ScoreMLP(args.hidden_dim).to(device) if args.method == "si_sde" else None

        params = list(velocity_model.parameters())
        if score_model is not None:
            params += list(score_model.parameters())
        optim = torch.optim.Adam(params, lr=args.lr)

        t0 = time.time()
        for i in range(args.iterations):
            optim.zero_grad()
            x_1 = checkerboard(args.batch_size, device)
            x_0 = torch.randn_like(x_1)

            loss_v = si_velocity_loss(velocity_model, x_0, x_1)

            if score_model is not None:
                loss_s = si_score_loss(score_model, x_0, x_1)
                loss = loss_v + loss_s
            else:
                loss = loss_v

            loss.backward()
            optim.step()

            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  iter {i+1:6d} | "
                    f"{elapsed*1000/args.print_every:5.1f} ms/step | "
                    f"loss {loss.item():.4f}"
                )
                t0 = time.time()

        ckpt = {"velocity": velocity_model.state_dict()}
        if score_model is not None:
            ckpt["score"] = score_model.state_dict()

        ckpt_path = os.path.join(args.ckpt_dir, f"{args.method}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved: {ckpt_path}")

    # ========================================================================
    # DSBM (Diffusion Schrödinger Bridge Matching, De Bortoli et al.)
    # ========================================================================
    elif args.method == "dsbm":
        from ot.dsbm import train_dsbm

        # Pre-generate source and target samples
        n_data = 50000
        data_x0 = torch.randn(n_data, 2)  # noise
        data_x1 = checkerboard(n_data)     # data

        net_fwd, net_bwd = train_dsbm(
            data_x0=data_x0,
            data_x1=data_x1,
            sigma=0.5,
            n_ipf=3,
            inner_iters=3000,
            batch_size=min(args.batch_size, 2048),
            lr=args.lr,
            d=2,
            hidden=256,
            device=device,
            verbose=True,
        )

        ckpt_path = os.path.join(args.ckpt_dir, "dsbm.pt")
        torch.save(
            {"fwd": net_fwd.state_dict(), "bwd": net_bwd.state_dict()},
            ckpt_path,
        )
        print(f"Saved: {ckpt_path}")

    # ========================================================================
    # Neural Flow Diffusion Models (Bartosh, Vetrov, Naesseth, NeurIPS 2024)
    # ========================================================================
    elif args.method == "nfdm":
        from ot.nfdm import NeuralDiffusion

        model = NeuralDiffusion(d=2).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        t0 = time.time()
        for i in range(args.iterations):
            optim.zero_grad()
            x = checkerboard(args.batch_size, device)
            t = torch.rand(x.shape[0], 1, device=device).clamp(1e-4, 1.0 - 1e-4)

            loss = model(x, t).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  iter {i+1:6d} | "
                    f"{elapsed*1000/args.print_every:5.1f} ms/step | "
                    f"loss {loss.item():.4f}"
                )
                t0 = time.time()

        ckpt_path = os.path.join(args.ckpt_dir, "nfdm.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
