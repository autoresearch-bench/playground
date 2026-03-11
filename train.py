"""
AlphaZero Chess — Training Pipeline

The loop:
  1. Self-play: generate games using MCTS + current network
  2. Train: update network on self-play data (state, policy, value)
  3. Repeat

Compute budget is tracked in GPU-seconds. The default budget (3600 GPU-sec)
corresponds to ~1 hour on an A100. Adjust in config.py or via CLI args.
"""

import os
import time
import random
import argparse
from collections import deque

import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import AlphaZeroConfig
from model import AlphaZeroNet
from mcts import run_mcts, board_to_tensor, move_to_action


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size buffer of (state, policy_target, value_target) tuples."""

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def push(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.stack([b[0] for b in batch])
        policies = np.stack([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)
        return (
            torch.tensor(states),
            torch.tensor(policies),
            torch.tensor(values).unsqueeze(1),
        )


# ---------------------------------------------------------------------------
# Self-Play
# ---------------------------------------------------------------------------

def self_play_game(network, cfg: AlphaZeroConfig):
    """
    Play one game of self-play using MCTS. Returns list of
    (board_tensor, action_probs, result_from_current_player) tuples.
    """
    board = chess.Board()
    history = []  # (state_tensor, policy, current_player)
    move_count = 0

    while not board.is_game_over() and move_count < 512:
        state = board_to_tensor(board)

        # Run MCTS
        action_probs, _ = run_mcts(board, network, cfg.mcts, device=cfg.device)

        # Temperature-based move selection
        if move_count < cfg.mcts.temperature_threshold:
            # Sample proportionally (with temperature=1)
            probs = action_probs
            total = probs.sum()
            if total > 0:
                probs = probs / total
                action = np.random.choice(len(probs), p=probs)
            else:
                action = random.choice(range(len(probs)))
        else:
            # Greedy (temperature→0)
            action = np.argmax(action_probs)

        history.append((state, action_probs, board.turn))

        # Find and play the move
        legal_moves = list(board.legal_moves)
        action_to_move_map = {move_to_action(m, board): m for m in legal_moves}
        move = action_to_move_map.get(action)
        if move is None:
            # Fallback: pick the legal move with highest MCTS probability
            best_action = max(action_to_move_map.keys(), key=lambda a: action_probs[a])
            move = action_to_move_map[best_action]

        board.push(move)
        move_count += 1

    # Determine game result
    result = board.result()
    if result == "1-0":
        game_value = 1.0   # white wins
    elif result == "0-1":
        game_value = -1.0  # black wins
    else:
        game_value = 0.0   # draw

    # Assign values from each position's player perspective
    training_data = []
    for state, policy, player in history:
        if player == chess.WHITE:
            value = game_value
        else:
            value = -game_value
        training_data.append((state, policy, value))

    return training_data


# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------

def train_step(network, optimizer, replay_buffer, cfg: AlphaZeroConfig):
    """One training step: sample batch, compute loss, update weights."""
    network.train()
    states, target_policies, target_values = replay_buffer.sample(cfg.train.batch_size)
    states = states.to(cfg.device)
    target_policies = target_policies.to(cfg.device)
    target_values = target_values.to(cfg.device)

    # Forward pass
    pred_policies, pred_values = network(states)

    # Policy loss: cross-entropy with MCTS visit counts as soft targets
    policy_loss = -(target_policies * F.log_softmax(pred_policies, dim=1)).sum(dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(pred_values, target_values)

    # Total loss
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'total_loss': loss.item(),
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(network, optimizer, iteration, path='checkpoint.pt'):
    torch.save({
        'iteration': iteration,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(network, optimizer, path='checkpoint.pt'):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location='cpu')
        network.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['iteration']
    return 0


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Chess Training')
    parser.add_argument('--gpu-seconds', type=float, default=None,
                        help='Total GPU-second budget per iteration (default: 3600)')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of self-play → train cycles')
    parser.add_argument('--games-per-iter', type=int, default=None,
                        help='Self-play games per iteration')
    parser.add_argument('--simulations', type=int, default=None,
                        help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    cfg = AlphaZeroConfig()
    cfg.device = args.device
    if args.gpu_seconds is not None:
        cfg.compute.total_gpu_seconds = args.gpu_seconds
    if args.iterations is not None:
        cfg.train.num_iterations = args.iterations
    if args.games_per_iter is not None:
        cfg.train.games_per_iteration = args.games_per_iter
    if args.simulations is not None:
        cfg.mcts.num_simulations = args.simulations

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device(cfg.device)
    network = AlphaZeroNet(cfg.model).to(device)
    optimizer = optim.SGD(
        network.parameters(),
        lr=cfg.train.learning_rate,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.train.lr_milestones, gamma=cfg.train.lr_gamma,
    )
    replay_buffer = ReplayBuffer(cfg.train.replay_buffer_size)

    num_params = sum(p.numel() for p in network.parameters())
    print(f"AlphaZero Chess")
    print(f"  Parameters:     {num_params / 1e6:.2f}M")
    print(f"  Res blocks:     {cfg.model.num_res_blocks}")
    print(f"  Filters:        {cfg.model.num_filters}")
    print(f"  MCTS sims:      {cfg.mcts.num_simulations}")
    print(f"  GPU budget:     {cfg.compute.total_gpu_seconds:.0f}s per iteration")
    print(f"  Device:         {device}")
    print()

    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(network, optimizer, args.resume)
        print(f"Resumed from iteration {start_iteration}")

    wall_start = time.time()

    for iteration in range(start_iteration, cfg.train.num_iterations):
        iter_start = time.time()
        gpu_time_used = 0.0

        # --- Phase 1: Self-Play ---
        print(f"=== Iteration {iteration + 1}/{cfg.train.num_iterations} ===")
        print(f"  Self-play: generating {cfg.train.games_per_iteration} games...")

        network.eval()
        games_played = 0
        positions_added = 0

        for game_idx in range(cfg.train.games_per_iteration):
            t0 = time.time()
            game_data = self_play_game(network, cfg)
            dt = time.time() - t0
            gpu_time_used += dt

            for state, policy, value in game_data:
                replay_buffer.push(state, policy, value)
                positions_added += 1

            games_played += 1

            if (game_idx + 1) % 10 == 0:
                print(f"    Game {game_idx + 1}/{cfg.train.games_per_iteration} "
                      f"({len(game_data)} positions, {dt:.1f}s) "
                      f"[GPU budget: {gpu_time_used:.0f}/{cfg.compute.self_play_gpu_seconds:.0f}s]")

            # Check compute budget
            if gpu_time_used >= cfg.compute.self_play_gpu_seconds:
                print(f"  Self-play budget exhausted after {games_played} games")
                break

            # Check wall clock
            if time.time() - wall_start >= cfg.compute.max_wall_clock:
                print(f"  Wall clock limit reached")
                break

        print(f"  Self-play done: {games_played} games, {positions_added} positions, "
              f"buffer size: {len(replay_buffer)}")

        # --- Phase 2: Training ---
        if len(replay_buffer) < cfg.train.min_replay_size:
            print(f"  Skipping training (buffer {len(replay_buffer)} < {cfg.train.min_replay_size})")
            continue

        print(f"  Training...")
        train_steps = 0
        train_gpu_time = 0.0
        total_loss_sum = 0.0

        while train_gpu_time < cfg.compute.training_gpu_seconds:
            t0 = time.time()
            losses = train_step(network, optimizer, replay_buffer, cfg)
            dt = time.time() - t0
            train_gpu_time += dt
            total_loss_sum += losses['total_loss']
            train_steps += 1

            if train_steps % 100 == 0:
                avg_loss = total_loss_sum / train_steps
                print(f"    Step {train_steps}: loss={avg_loss:.4f} "
                      f"(policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}) "
                      f"[GPU: {train_gpu_time:.0f}/{cfg.compute.training_gpu_seconds:.0f}s]")

            if time.time() - wall_start >= cfg.compute.max_wall_clock:
                break

        scheduler.step()
        avg_loss = total_loss_sum / max(train_steps, 1)
        iter_time = time.time() - iter_start

        print(f"  Training done: {train_steps} steps, avg_loss={avg_loss:.4f}")
        print(f"  Iteration time: {iter_time:.1f}s "
              f"(self-play: {gpu_time_used:.0f}s, train: {train_gpu_time:.0f}s)")
        print()

        # Checkpoint
        if (iteration + 1) % cfg.train.checkpoint_interval == 0 or \
           iteration == cfg.train.num_iterations - 1:
            save_checkpoint(network, optimizer, iteration + 1)
            print(f"  Checkpoint saved (iteration {iteration + 1})")

        # Wall clock check
        if time.time() - wall_start >= cfg.compute.max_wall_clock:
            print(f"Wall clock limit reached ({cfg.compute.max_wall_clock}s). Stopping.")
            break

    total_time = time.time() - wall_start
    print(f"\n--- Training Complete ---")
    print(f"Total wall time: {total_time:.1f}s")
    print(f"Final buffer size: {len(replay_buffer)}")
    save_checkpoint(network, optimizer, cfg.train.num_iterations, 'checkpoint_final.pt')
    print(f"Final checkpoint saved.")


if __name__ == '__main__':
    main()
