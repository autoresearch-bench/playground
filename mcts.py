"""
AlphaZero Chess — Monte Carlo Tree Search

Standard PUCT-based MCTS as described in the AlphaZero paper.
Each node stores visit counts N, total value W, prior probability P,
and mean value Q = W/N.

Move encoding: AlphaZero uses 8×8×73 = 4672 possible actions.
  - Indices 0-55:  queen moves (7 distances × 8 directions)
  - Indices 56-63: knight moves (8 possible L-shapes)
  - Indices 64-72: underpromotions (3 piece types × 3 directions)
"""

import math
import numpy as np
import chess

from config import MCTSConfig


# ---------------------------------------------------------------------------
# Move ↔ action index mapping
# ---------------------------------------------------------------------------

# Direction vectors for queen-type moves (N, NE, E, SE, S, SW, W, NW)
QUEEN_DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def move_to_action(move: chess.Move, board: chess.Board) -> int:
    """Convert a python-chess Move to an action index in [0, 4672)."""
    from_sq = move.from_square
    to_sq = move.to_square
    from_row, from_col = divmod(from_sq, 8)
    to_row, to_col = divmod(to_sq, 8)

    # Flip perspective if black to move (AlphaZero always encodes from current player's view)
    if board.turn == chess.BLACK:
        from_row = 7 - from_row
        to_row = 7 - to_row

    dr = to_row - from_row
    dc = to_col - from_col

    # Underpromotion
    if move.promotion and move.promotion != chess.QUEEN:
        promo_idx = {chess.ROOK: 0, chess.BISHOP: 1, chess.KNIGHT: 2}[move.promotion]
        if dc == -1:
            direction = 0
        elif dc == 0:
            direction = 1
        else:
            direction = 2
        plane = 64 + promo_idx * 3 + direction
        return from_row * 8 * 73 + from_col * 73 + plane

    # Knight move
    if (dr, dc) in KNIGHT_MOVES:
        knight_idx = KNIGHT_MOVES.index((dr, dc))
        plane = 56 + knight_idx
        return from_row * 8 * 73 + from_col * 73 + plane

    # Queen-type move (includes queen promotions treated as normal moves)
    distance = max(abs(dr), abs(dc))
    direction_vec = (
        (dr // distance if dr != 0 else 0),
        (dc // distance if dc != 0 else 0),
    )
    if direction_vec in QUEEN_DIRS:
        dir_idx = QUEEN_DIRS.index(direction_vec)
        plane = dir_idx * 7 + (distance - 1)
        return from_row * 8 * 73 + from_col * 73 + plane

    # Fallback (shouldn't happen for legal moves)
    return 0


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Convert action index back to a python-chess Move. Returns None if illegal."""
    from_row = action // (8 * 73)
    remainder = action % (8 * 73)
    from_col = remainder // 73
    plane = remainder % 73

    if plane < 56:
        # Queen move
        dir_idx = plane // 7
        distance = (plane % 7) + 1
        dr, dc = QUEEN_DIRS[dir_idx]
        to_row = from_row + dr * distance
        to_col = from_col + dc * distance
        promotion = None
        # Check if this is a pawn reaching last rank → auto-promote to queen
    elif plane < 64:
        # Knight move
        knight_idx = plane - 56
        dr, dc = KNIGHT_MOVES[knight_idx]
        to_row = from_row + dr
        to_col = from_col + dc
        promotion = None
    else:
        # Underpromotion
        promo_plane = plane - 64
        promo_idx = promo_plane // 3
        direction = promo_plane % 3
        promotion = [chess.ROOK, chess.BISHOP, chess.KNIGHT][promo_idx]
        to_row = from_row + 1  # pawn moves forward
        to_col = from_col + (direction - 1)  # -1, 0, or +1

    # Flip back if black
    if board.turn == chess.BLACK:
        from_row = 7 - from_row
        to_row = 7 - to_row

    if not (0 <= to_row < 8 and 0 <= to_col < 8):
        return None

    from_sq = from_row * 8 + from_col
    to_sq = to_row * 8 + to_col

    # Auto-promote pawns to queen if reaching last rank
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            target_rank = 7 if board.turn == chess.WHITE else 0
            if to_row == target_rank:
                promotion = chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if move in board.legal_moves:
        return move
    return None


# ---------------------------------------------------------------------------
# Board → tensor encoding
# ---------------------------------------------------------------------------

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board as an (18, 8, 8) float32 numpy array."""
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    # Piece planes (0-11)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            row, col = divmod(sq, 8)
            plane_idx = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
            planes[plane_idx, row, col] = 1.0

    # Current player (plane 12)
    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    # Castling rights (planes 13-16)
    planes[13, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[15, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[16, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    # En passant (plane 17)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        planes[17, row, col] = 1.0

    return planes


# ---------------------------------------------------------------------------
# MCTS Node and Search
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ['parent', 'action', 'prior', 'visit_count', 'total_value',
                 'children', 'board', 'is_expanded']

    def __init__(self, board: chess.Board, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children = []
        self.board = board
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


def select_child(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Select child with highest PUCT score."""
    sqrt_parent = math.sqrt(node.visit_count)
    best_score = -float('inf')
    best_child = None

    for child in node.children:
        # PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        exploit = child.q_value
        explore = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        score = exploit + explore
        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def expand_node(node: MCTSNode, policy_probs: np.ndarray):
    """Expand node by creating children for all legal moves."""
    board = node.board
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return

    # Get action indices for legal moves and their priors
    action_priors = []
    for move in legal_moves:
        action_idx = move_to_action(move, board)
        prior = policy_probs[action_idx]
        action_priors.append((move, action_idx, prior))

    # Renormalize priors over legal moves
    total_prior = sum(p for _, _, p in action_priors)
    if total_prior > 0:
        action_priors = [(m, a, p / total_prior) for m, a, p in action_priors]
    else:
        uniform = 1.0 / len(action_priors)
        action_priors = [(m, a, uniform) for m, a, _ in action_priors]

    for move, action_idx, prior in action_priors:
        child_board = board.copy()
        child_board.push(move)
        child = MCTSNode(child_board, parent=node, action=action_idx, prior=prior)
        node.children.append(child)

    node.is_expanded = True


def backpropagate(node: MCTSNode, value: float):
    """Propagate value up the tree, flipping sign at each level."""
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        value = -value  # opponent's perspective
        node = node.parent


def run_mcts(board: chess.Board, network, cfg: MCTSConfig, device='cuda'):
    """
    Run MCTS from the given position. Returns (action_probs, root_value).
    action_probs is a numpy array of shape (4672,) with visit-count-based probabilities.
    """
    import torch

    root = MCTSNode(board.copy())

    # Get network prediction for root
    board_tensor = torch.tensor(board_to_tensor(board), device=device).unsqueeze(0)
    policy_probs, root_value = network.predict(board_tensor.to(device))
    policy_probs = policy_probs.cpu().numpy()
    expand_node(root, policy_probs)

    # Add Dirichlet noise at root for exploration
    if root.children:
        noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(root.children))
        for child, n in zip(root.children, noise):
            child.prior = (1 - cfg.dirichlet_epsilon) * child.prior + cfg.dirichlet_epsilon * n

    # Run simulations
    for _ in range(cfg.num_simulations):
        node = root

        # Selection: traverse tree using PUCT
        while node.is_expanded and node.children:
            node = select_child(node, cfg.c_puct)

        # Check terminal
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":
                value = 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                value = 1.0 if node.board.turn == chess.WHITE else -1.0
            else:
                value = 0.0
        else:
            # Expansion + evaluation
            bt = torch.tensor(board_to_tensor(node.board), device=device).unsqueeze(0)
            p, v = network.predict(bt.to(device))
            policy_probs = p.cpu().numpy()
            value = v
            expand_node(node, policy_probs)

        # Backpropagation
        backpropagate(node, value)

    # Extract action probabilities from visit counts
    action_probs = np.zeros(4672, dtype=np.float32)
    for child in root.children:
        action_probs[child.action] = child.visit_count

    total_visits = action_probs.sum()
    if total_visits > 0:
        action_probs /= total_visits

    return action_probs, root.q_value
