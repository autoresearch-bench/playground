"""
One-time data preparation for the Genome LM Speedrun.

Generates synthetic human-genome-like sequences (dry run) with realistic
nucleotide composition and downstream task datasets for three genomic
classification benchmarks (NucleL-style evaluation).

Usage:
    python prepare.py                         # 100k synthetic train + 5k val
    python prepare.py --num-train 200000      # larger synthetic set
    python prepare.py --seed 123              # different random seed

Data is stored in ~/.cache/genome-lm-speedrun/.

Downstream tasks (NucleL-style):
    1. Promoter detection     — binary,  detect core promoter elements
    2. Splice site detection  — 3-class, donor / acceptor / background
    3. CpG island detection   — binary,  CpG-enriched vs background regions
"""

import os
import argparse
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN    = 512     # context window in tokens (includes <cls> and <eos>)
MAX_GENOMIC_LEN = 510    # effective genomic window (MAX_SEQ_LEN - 2 special tokens)
TIME_BUDGET    = 300     # training time budget in seconds (5 min)
VOCAB_SIZE     = 10      # nucleotide vocabulary size
MLM_MASK_PROB  = 0.15    # fraction of tokens selected for masking
EVAL_BATCHES   = 100     # number of batches in the fixed validation evaluation

# Downstream task settings
N_DOWNSTREAM_TRAIN      = 5_000  # samples per class per task (train)
N_DOWNSTREAM_VAL        = 1_000  # samples per class per task (val)
DOWNSTREAM_SEQ_LEN      = MAX_GENOMIC_LEN   # nucleotides per downstream sample
DOWNSTREAM_PROBE_STEPS  = 200    # linear-probe training steps

# ---------------------------------------------------------------------------
# Vocabulary (10 tokens — do not modify)
# ---------------------------------------------------------------------------

VOCAB = {
    "<cls>":  0,
    "<pad>":  1,
    "<eos>":  2,
    "<unk>":  3,
    "A":      4,
    "T":      5,
    "G":      6,
    "C":      7,
    "N":      8,
    "<mask>": 9,
}
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

CLS_TOKEN  = VOCAB["<cls>"]   # 0
PAD_TOKEN  = VOCAB["<pad>"]   # 1
EOS_TOKEN  = VOCAB["<eos>"]   # 2
UNK_TOKEN  = VOCAB["<unk>"]   # 3
MASK_TOKEN = VOCAB["<mask>"]  # 9

# Nucleotide token IDs — used for random-token replacement in MLM
_NT_TOKEN_IDS = np.array([VOCAB["A"], VOCAB["T"], VOCAB["G"], VOCAB["C"]], dtype=np.int64)

# Human genome (hg38) approximate base frequencies
_HG38_FREQ = np.array([0.295, 0.295, 0.205, 0.205])  # A, T, G, C

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "genome-lm-speedrun")
DATA_DIR  = os.path.join(CACHE_DIR, "data")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal DNA/RNA sequence tokenizer with a fixed 10-token vocabulary."""

    def encode(self, seq: str) -> list:
        """Encode a nucleotide sequence to token IDs (prepends <cls>, appends <eos>)."""
        tokens = [CLS_TOKEN]
        for nt in seq.upper():
            tokens.append(VOCAB.get(nt, UNK_TOKEN))
        tokens.append(EOS_TOKEN)
        return tokens

    def decode(self, ids) -> str:
        """Decode token IDs back to nucleotide string (strips special tokens)."""
        skip = {CLS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN, MASK_TOKEN}
        return "".join(ID_TO_TOKEN.get(i, "?") for i in ids if i not in skip)

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE


# ---------------------------------------------------------------------------
# Sequence generation helpers
# ---------------------------------------------------------------------------

def _random_seq(length: int, rng: np.random.Generator,
                freq=None) -> str:
    """Random nucleotide sequence with given base frequencies."""
    if freq is None:
        freq = _HG38_FREQ
    return "".join(rng.choice(list("ATGC"), size=length, p=freq))


def _inject_motif(seq: str, motif: str, pos: int,
                  noise_rate: float = 0.1, rng=None) -> str:
    """Inject a motif at pos with optional point-mutation noise."""
    seq = list(seq)
    for i, base in enumerate(motif):
        p = pos + i
        if p >= len(seq):
            break
        if rng is not None and rng.random() < noise_rate:
            seq[p] = rng.choice(list("ATGC"))
        else:
            seq[p] = base
    return "".join(seq)


# ---------------------------------------------------------------------------
# Downstream task 1: Core Promoter Detection (binary)
# ---------------------------------------------------------------------------
# Positive: TATA-box (~−30) + Initiator element (~+1) + elevated upstream GC
# Negative: background hg38 composition, no canonical promoter motif

def _gen_promoter(n: int, label: int, rng: np.random.Generator) -> tuple:
    seqs, labels = [], []
    center = DOWNSTREAM_SEQ_LEN // 2

    for _ in range(n):
        seq = list(_random_seq(DOWNSTREAM_SEQ_LEN, rng))

        if label == 1:
            # TATA box variants at ~−30 from TSS
            tata = rng.choice(["TATAAA", "TATATAT", "TATATA", "TATAAG"])
            tata_pos = center - 35 + int(rng.integers(-5, 6))
            tata_pos = max(0, min(tata_pos, DOWNSTREAM_SEQ_LEN - len(tata)))
            seq = list(_inject_motif("".join(seq), tata, tata_pos,
                                     noise_rate=0.10, rng=rng))

            # Initiator (Inr) element at TSS
            inr = rng.choice(["TCAGT", "TCATT", "TCACT", "TCANTYY"])[:5]
            inr_pos = center + int(rng.integers(-2, 3))
            inr_pos = max(0, min(inr_pos, DOWNSTREAM_SEQ_LEN - len(inr)))
            seq = list(_inject_motif("".join(seq), inr, inr_pos,
                                     noise_rate=0.15, rng=rng))

            # CpG-like enrichment in −100 to −50 window
            for i in range(max(0, center - 100), max(0, center - 50)):
                if rng.random() < 0.30:
                    seq[i] = rng.choice(["G", "C"])

        seqs.append("".join(seq))
        labels.append(label)

    return seqs, labels


# ---------------------------------------------------------------------------
# Downstream task 2: Splice Site Detection (3-class)
# ---------------------------------------------------------------------------
# Label 0 — Donor:   GT at center, GC-rich exon upstream, AT-rich intron downstream
# Label 1 — Acceptor: AG at center, polypyrimidine tract + branch point upstream
# Label 2 — Background: no canonical dinucleotide signal

def _gen_splice(n: int, label: int, rng: np.random.Generator) -> tuple:
    seqs, labels = [], []
    center = DOWNSTREAM_SEQ_LEN // 2

    for _ in range(n):
        seq = list(_random_seq(DOWNSTREAM_SEQ_LEN, rng))

        if label == 0:  # Donor
            # Upstream exon: slightly GC-rich
            for i in range(center):
                if rng.random() < 0.30:
                    seq[i] = rng.choice(["G", "C"])
            # GT dinucleotide at splice junction
            if rng.random() > 0.05:
                seq[center]     = "G"
                seq[center + 1] = "T"
            # Downstream intron: AT-rich
            for i in range(center + 2, DOWNSTREAM_SEQ_LEN):
                if rng.random() < 0.25:
                    seq[i] = rng.choice(["A", "T"])

        elif label == 1:  # Acceptor
            # Polypyrimidine tract −30 to −5
            pp_s = max(0, center - 30)
            pp_e = max(0, center - 5)
            for i in range(pp_s, pp_e):
                if rng.random() < 0.70:
                    seq[i] = rng.choice(["C", "T"])
            # Branch-point CTRAY ~−25
            bp_pos = center - 25 + int(rng.integers(-5, 6))
            bp_pos = max(0, min(bp_pos, DOWNSTREAM_SEQ_LEN - 5))
            bp_motif = "CT" + rng.choice(["A"]) + rng.choice(["A", "G"]) + rng.choice(list("ATGC"))
            seq = list(_inject_motif("".join(seq), bp_motif, bp_pos,
                                     noise_rate=0.20, rng=rng))
            # AG dinucleotide at acceptor
            if rng.random() > 0.05:
                seq[center - 1] = "A"
                seq[center]     = "G"

        # label == 2: pure background, no changes needed

        seqs.append("".join(seq))
        labels.append(label)

    return seqs, labels


# ---------------------------------------------------------------------------
# Downstream task 3: CpG Island Detection (binary)
# ---------------------------------------------------------------------------
# Positive: GC content 55–70%, enriched CpG dinucleotides (obs/exp CpG > 0.6)
# Negative: AT-rich background, CpG-depleted (methylation-driven depletion)

def _gen_cpg(n: int, label: int, rng: np.random.Generator) -> tuple:
    seqs, labels = [], []

    for _ in range(n):
        if label == 1:
            gc = 0.55 + rng.random() * 0.15          # 55–70 % GC
            g_f = c_f = gc / 2
            at_f = (1 - gc) / 2
            seq = list(_random_seq(DOWNSTREAM_SEQ_LEN, rng,
                                   np.array([at_f, at_f, g_f, c_f])))
            # Enrich CpG dinucleotides
            for i in range(DOWNSTREAM_SEQ_LEN - 1):
                if rng.random() < 0.15:
                    seq[i]     = "C"
                    seq[i + 1] = "G"
        else:
            # AT-rich, CpG-depleted
            freq = np.array([0.30, 0.30, 0.20, 0.20])
            seq = list(_random_seq(DOWNSTREAM_SEQ_LEN, rng, freq))
            for i in range(DOWNSTREAM_SEQ_LEN - 1):
                if seq[i] == "C" and seq[i + 1] == "G" and rng.random() < 0.80:
                    if rng.random() < 0.5:
                        seq[i] = "T"      # TG (C→T deamination)
                    else:
                        seq[i + 1] = "A"  # CA

        seqs.append("".join(seq))
        labels.append(label)

    return seqs, labels


# ---------------------------------------------------------------------------
# Pretraining sequence generation
# ---------------------------------------------------------------------------

def _generate_genomic_sequences(n: int, rng: np.random.Generator) -> list:
    """Generate n random genomic sequences with hg38-like base composition."""
    seqs = []
    for _ in range(n):
        length = int(rng.integers(50, MAX_GENOMIC_LEN + 1))
        seqs.append(_random_seq(length, rng))
    return seqs


def _apply_mlm(ids: np.ndarray, rng: np.random.Generator) -> tuple:
    """
    BERT-style masked language modeling on a batch of token ID arrays.

    Args:
        ids: int64 array (B, T) — padded token sequences
        rng: NumPy random generator

    Returns:
        masked: int64 array (B, T) — 15 % eligible tokens replaced
        labels: int64 array (B, T) — true IDs at masked positions, -100 elsewhere

    Masking strategy (identical to BERT):
        80 % → <mask>
        10 % → random nucleotide
        10 % → unchanged (kept as training signal)
    """
    B, T = ids.shape
    masked = ids.copy()
    labels = np.full((B, T), -100, dtype=np.int64)

    special_ids = {CLS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    eligible = ~np.isin(ids, list(special_ids))

    for b in range(B):
        pos = np.where(eligible[b])[0]
        if len(pos) == 0:
            continue
        n_mask = max(1, int(round(len(pos) * MLM_MASK_PROB)))
        chosen = rng.choice(pos, size=n_mask, replace=False)
        labels[b, chosen] = ids[b, chosen]

        r = rng.random(size=n_mask)
        for k, p in enumerate(chosen):
            if r[k] < 0.80:
                masked[b, p] = MASK_TOKEN
            elif r[k] < 0.90:
                masked[b, p] = int(rng.choice(_NT_TOKEN_IDS))
            # else: keep original (10 %)

    return masked, labels


# ---------------------------------------------------------------------------
# Data persistence helpers
# ---------------------------------------------------------------------------

def _save_pretrain_split(seqs: list, split: str, tokenizer: Tokenizer):
    """Tokenize sequences and save as flat uint8 binary + int64 offset array."""
    os.makedirs(DATA_DIR, exist_ok=True)

    all_ids: list[int] = []
    offsets: list[int] = [0]

    for seq in seqs:
        ids = tokenizer.encode(seq[:MAX_GENOMIC_LEN])
        all_ids.extend(ids)
        offsets.append(len(all_ids))

    data_arr = np.array(all_ids, dtype=np.uint8)
    offs_arr = np.array(offsets, dtype=np.int64)

    data_arr.tofile(os.path.join(DATA_DIR, f"{split}_data.bin"))
    np.save(os.path.join(DATA_DIR, f"{split}_offsets.npy"), offs_arr)

    print(f"  {split:5s}: {len(seqs):,} seqs, {len(all_ids):,} tokens")


def _pack_downstream(seqs: list, labels: list, tokenizer: Tokenizer) -> tuple:
    """Pack downstream sequences into a padded int32 token array."""
    n   = len(seqs)
    T   = DOWNSTREAM_SEQ_LEN + 2    # +CLS +EOS
    X   = np.full((n, T), PAD_TOKEN, dtype=np.int32)
    y   = np.array(labels, dtype=np.int64)

    for i, seq in enumerate(seqs):
        ids = tokenizer.encode(seq[:DOWNSTREAM_SEQ_LEN])
        X[i, :len(ids)] = ids

    return X, y


def _save_downstream(task: str, split: str,
                     seqs: list, labels: list, tokenizer: Tokenizer):
    os.makedirs(DATA_DIR, exist_ok=True)
    X, y = _pack_downstream(seqs, labels, tokenizer)
    np.save(os.path.join(DATA_DIR, f"{task}_{split}_X.npy"), X)
    np.save(os.path.join(DATA_DIR, f"{task}_{split}_y.npy"), y)


# ---------------------------------------------------------------------------
# prepare_data — top-level entry point
# ---------------------------------------------------------------------------

def prepare_data(
    n_train:              int = 100_000,
    n_val:                int = 5_000,
    seed:                 int = 42,
    n_downstream_train:   int = N_DOWNSTREAM_TRAIN,
    n_downstream_val:     int = N_DOWNSTREAM_VAL,
):
    """
    Prepare all data for the Genome LM Speedrun.

    Writes to DATA_DIR (~/.cache/genome-lm-speedrun/data/):
        train_data.bin, train_offsets.npy        — pretraining train split
        val_data.bin,   val_offsets.npy          — pretraining val split
        promoter_{train,val}_{X,y}.npy           — Task 1: promoter detection
        splice_{train,val}_{X,y}.npy             — Task 2: splice site detection
        cpg_{train,val}_{X,y}.npy                — Task 3: CpG island detection
    """
    rng = np.random.default_rng(seed)
    tok = Tokenizer()

    # ---- Pretraining data --------------------------------------------------
    print(f"Generating {n_train:,} train + {n_val:,} val genomic sequences ...")
    _save_pretrain_split(_generate_genomic_sequences(n_train, rng), "train", tok)
    _save_pretrain_split(_generate_genomic_sequences(n_val,   rng), "val",   tok)

    # ---- Downstream tasks --------------------------------------------------
    print("Generating downstream task data ...")
    for split, n, rng_seed in [
        ("train", n_downstream_train, seed + 100),
        ("val",   n_downstream_val,   seed + 200),
    ]:
        ds_rng = np.random.default_rng(rng_seed)

        # Task 1: Promoter detection (binary)
        pos_s, pos_l = _gen_promoter(n, 1, ds_rng)
        neg_s, neg_l = _gen_promoter(n, 0, ds_rng)
        _save_downstream("promoter", split, pos_s + neg_s, pos_l + neg_l, tok)
        print(f"  promoter {split}: {2*n:,} ({n:,} pos + {n:,} neg)")

        # Task 2: Splice site detection (3-class)
        don_s, don_l = _gen_splice(n, 0, ds_rng)
        acc_s, acc_l = _gen_splice(n, 1, ds_rng)
        bg_s,  bg_l  = _gen_splice(n, 2, ds_rng)
        _save_downstream("splice", split,
                         don_s + acc_s + bg_s,
                         don_l + acc_l + bg_l, tok)
        print(f"  splice   {split}: {3*n:,} ({n:,} donor + {n:,} acceptor + {n:,} bg)")

        # Task 3: CpG island detection (binary)
        cpg_s,  cpg_l  = _gen_cpg(n, 1, ds_rng)
        ncpg_s, ncpg_l = _gen_cpg(n, 0, ds_rng)
        _save_downstream("cpg", split, cpg_s + ncpg_s, cpg_l + ncpg_l, tok)
        print(f"  cpg      {split}: {2*n:,} ({n:,} island + {n:,} non-island)")

    print(f"All data written to {DATA_DIR}")


# ---------------------------------------------------------------------------
# Pretraining dataloader
# ---------------------------------------------------------------------------

def _ensure_data():
    train_bin = os.path.join(DATA_DIR, "train_data.bin")
    if not os.path.exists(train_bin):
        print("Data not found — running prepare_data() ...")
        prepare_data()


def make_dataloader(batch_size: int, seq_len: int, split: str):
    """
    Infinite generator yielding MLM batches for pretraining.

    Yields:
        input_ids : (B, seq_len) int64 CUDA tensor — masked token IDs
        labels    : (B, seq_len) int64 CUDA tensor — true IDs at masked pos, -100 elsewhere
        attn_mask : (B, seq_len) bool  CUDA tensor — True for real tokens, False for PAD
    """
    _ensure_data()

    data    = np.fromfile(os.path.join(DATA_DIR, f"{split}_data.bin"), dtype=np.uint8)
    offsets = np.load(os.path.join(DATA_DIR, f"{split}_offsets.npy"))
    n_seqs  = len(offsets) - 1

    rng = np.random.default_rng(0 if split == "train" else 1)

    while True:
        idx = rng.integers(0, n_seqs, size=batch_size)
        batch = np.full((batch_size, seq_len), PAD_TOKEN, dtype=np.int64)

        for i, si in enumerate(idx):
            s, e   = int(offsets[si]), int(offsets[si + 1])
            seq    = data[s:e].astype(np.int64)
            L      = min(len(seq), seq_len)
            batch[i, :L] = seq[:L]

        masked, labels = _apply_mlm(batch, rng)
        attn_mask = batch != PAD_TOKEN

        yield (
            torch.from_numpy(masked).cuda(),
            torch.from_numpy(labels).cuda(),
            torch.from_numpy(attn_mask).cuda(),
        )


# ---------------------------------------------------------------------------
# Pretraining evaluation
# ---------------------------------------------------------------------------

def evaluate_mlm_loss(model, batch_size: int) -> float:
    """
    Evaluate masked-LM cross-entropy loss on the held-out validation set.

    Model interface:
        model(input_ids, labels, reduction='none') → flat 1-D tensor of
        per-token cross-entropy losses (0.0 at unmasked positions).
    """
    loader = make_dataloader(batch_size, MAX_SEQ_LEN, "val")

    total_loss = 0.0
    total_toks = 0

    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y, _ = next(loader)
            loss = model(x, y, reduction="none")
            mask  = y.view(-1) != -100
            total_loss += loss[mask].sum().item()
            total_toks += mask.sum().item()

    return total_loss / max(total_toks, 1)


# ---------------------------------------------------------------------------
# Downstream evaluation — linear probing
# ---------------------------------------------------------------------------

def evaluate_downstream(model, batch_size: int = 64) -> dict:
    """
    Evaluate pretrained representations on three NucleL-style genomic tasks
    via frozen-encoder linear probing.

    Model interface:
        model.encode(input_ids) → (B, D) float tensor of CLS-token representations

    Returns dict with keys: promoter_acc, splice_acc, cpg_acc.
    If the model does not implement `.encode()`, returns an empty dict.
    """
    if not hasattr(model, "encode"):
        return {}

    results = {}
    for task, n_cls in [("promoter", 2), ("splice", 3), ("cpg", 2)]:
        try:
            acc = _linear_probe(model, task, n_cls, batch_size)
            results[f"{task}_acc"] = acc
            print(f"  {task:8s} linear probe acc: {acc:.4f}")
        except Exception as exc:
            print(f"  {task} downstream eval failed: {exc}")
            results[f"{task}_acc"] = 0.0

    return results


def _linear_probe(model, task: str, n_classes: int, batch_size: int) -> float:
    """Extract CLS representations, train a linear head, return val accuracy."""
    _ensure_data()

    X_tr = np.load(os.path.join(DATA_DIR, f"{task}_train_X.npy"))
    y_tr = np.load(os.path.join(DATA_DIR, f"{task}_train_y.npy"))
    X_va = np.load(os.path.join(DATA_DIR, f"{task}_val_X.npy"))
    y_va = np.load(os.path.join(DATA_DIR, f"{task}_val_y.npy"))

    def _extract(X: np.ndarray) -> torch.Tensor:
        parts = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.from_numpy(X[i:i + batch_size]).long().cuda()
                parts.append(model.encode(xb).cpu())
        return torch.cat(parts, dim=0)

    tr_repr = _extract(X_tr)   # (N, D)
    va_repr = _extract(X_va)

    D     = tr_repr.shape[1]
    probe = torch.nn.Linear(D, n_classes).cuda()
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    tr_r  = tr_repr.cuda()
    tr_y  = torch.from_numpy(y_tr).long().cuda()
    N     = len(tr_y)

    probe.train()
    rng = np.random.default_rng(42)
    for _ in range(DOWNSTREAM_PROBE_STEPS):
        idx = torch.from_numpy(rng.integers(0, N, size=batch_size)).long()
        loss = torch.nn.functional.cross_entropy(probe(tr_r[idx]), tr_y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(va_repr.cuda()).argmax(dim=-1)
        acc   = (preds == torch.from_numpy(y_va).long().cuda()).float().mean().item()

    return acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Genome LM Speedrun data")
    parser.add_argument("--num-train",            type=int, default=100_000)
    parser.add_argument("--num-val",              type=int, default=5_000)
    parser.add_argument("--seed",                 type=int, default=42)
    parser.add_argument("--n-downstream-train",   type=int, default=N_DOWNSTREAM_TRAIN)
    parser.add_argument("--n-downstream-val",     type=int, default=N_DOWNSTREAM_VAL)
    args = parser.parse_args()

    t0 = time.time()
    prepare_data(
        n_train=args.num_train,
        n_val=args.num_val,
        seed=args.seed,
        n_downstream_train=args.n_downstream_train,
        n_downstream_val=args.n_downstream_val,
    )
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
