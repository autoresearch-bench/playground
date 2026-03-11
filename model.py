"""
AlphaZero Chess — Neural Network

ResNet-style architecture with dual heads:
  - Policy head: probability distribution over all possible moves (4672 outputs)
  - Value head: scalar evaluation of the position [-1, 1]

Board encoding (18 input planes on 8×8):
  Planes 0-5:   white pieces (pawn, knight, bishop, rook, queen, king)
  Planes 6-11:  black pieces (pawn, knight, bishop, rook, queen, king)
  Plane 12:     current player (all 1s if white to move, all 0s if black)
  Planes 13-16: castling rights (white kingside, white queenside, black kingside, black queenside)
  Plane 17:     en passant square (1 at the target square, 0 elsewhere)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class ResBlock(nn.Module):
    """Residual block: conv → BN → ReLU → conv → BN, then skip + ReLU."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style neural network.

    Input:  (B, 18, 8, 8) board tensor
    Output: policy logits (B, 4672), value (B, 1)
    """

    def __init__(self, cfg: ModelConfig = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        nf = cfg.num_filters

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(cfg.input_planes, nf, 3, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
        )

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock(nf) for _ in range(cfg.num_res_blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(nf, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, cfg.policy_output_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(nf, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """Returns (policy_logits, value) where value is in [-1, 1]."""
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def predict(self, board_tensor):
        """Single-board inference. Returns (policy_probs, value_scalar)."""
        self.eval()
        with torch.no_grad():
            if board_tensor.dim() == 3:
                board_tensor = board_tensor.unsqueeze(0)
            policy_logits, value = self(board_tensor)
            policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs.squeeze(0), value.item()
