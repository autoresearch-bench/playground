"""
AlphaZero Chess — Configuration & Compute Budget

Resource management is expressed in GPU-seconds. One "compute unit" = 1 second
of GPU time on the reference device (A100-80GB). All budgets scale linearly:
if you have a faster GPU, you finish sooner; slower GPU, takes longer.

Default budget: 3600 GPU-seconds (1 hour A100 equivalent) per training iteration.
"""

from dataclasses import dataclass, field


@dataclass
class ComputeBudget:
    """Controls how much compute to spend on each phase."""

    # Total GPU-seconds budget per training iteration
    total_gpu_seconds: float = 3600.0

    # Fraction of budget allocated to self-play vs training
    self_play_fraction: float = 0.75  # 75% self-play, 25% training

    # Maximum wall-clock time (seconds) as a hard stop, regardless of GPU speed
    max_wall_clock: float = 7200.0  # 2 hours

    @property
    def self_play_gpu_seconds(self) -> float:
        return self.total_gpu_seconds * self.self_play_fraction

    @property
    def training_gpu_seconds(self) -> float:
        return self.total_gpu_seconds * (1 - self.self_play_fraction)


@dataclass
class ModelConfig:
    """Neural network architecture."""
    num_res_blocks: int = 10
    num_filters: int = 128
    input_planes: int = 18     # 12 piece planes + 1 color + 4 castling + 1 en passant
    policy_output_size: int = 4672  # 73 move types × 64 squares (8×8)
    board_size: int = 8


@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search parameters."""
    num_simulations: int = 400
    c_puct: float = 1.5        # exploration constant
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25  # root noise mixing weight
    temperature_threshold: int = 30  # move number after which temperature drops to 0


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    batch_size: int = 256
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_milestones: list = field(default_factory=lambda: [100, 200, 300])
    lr_gamma: float = 0.1
    replay_buffer_size: int = 200_000
    min_replay_size: int = 10_000   # minimum samples before training starts
    checkpoint_interval: int = 100   # save every N training steps
    num_iterations: int = 100        # number of self-play → train cycles
    games_per_iteration: int = 100


@dataclass
class AlphaZeroConfig:
    """Top-level configuration."""
    compute: ComputeBudget = field(default_factory=ComputeBudget)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
