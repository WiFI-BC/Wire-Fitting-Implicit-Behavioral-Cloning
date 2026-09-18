"""Precompute language-goal embeddings for the LIBERO-Goal suite.

Run ONCE (after installing the LIBERO package + the libero_goal demos) to
produce the embedding cache consumed by `LiberoGoalDataset` and the eval
simulation. The cache is keyed by task NAME and stored in benchmark order.

Usage
-----
    uv run --managed-python --extra libero python scripts/precompute_libero_goal_embs.py \
        --out datasets/libero/libero_goal_goal_embs.npz --encoder minilm

Encoders
--------
  minilm : sentence-transformers `all-MiniLM-L6-v2` (384-D). Default; light.
  bert   : HuggingFace `bert-base-uncased`, mean-pooled (768-D). Matches the
           encoder family LIBERO's own `get_task_embs` uses.

The script prints the embedding dim AND, if it can open the first demo file,
the resulting libero_goal `state_dim` (= low-dim obs dim + embedding dim) so
you can drop it straight into config/config.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `utils` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.libero import (  # noqa: E402
    get_task_infos,
    save_goal_embeddings,
    select_lowdim_obs_keys,
)


def encode_minilm(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(model.encode(texts, convert_to_numpy=True), dtype=np.float32)


def encode_bert(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").eval()
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state  # (B, T, 768)
    mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # mean-pool
    return pooled.cpu().numpy().astype(np.float32)


ENCODERS = {"minilm": encode_minilm, "bert": encode_bert}


def probe_obs_dim(demo_file: str) -> int | None:
    """Return the low-dim obs vector length of the first demo, or None."""
    try:
        import h5py

        with h5py.File(demo_file, "r") as f:
            data = f["data"]
            dk = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))[0]
            obs_grp = data[dk]["obs"]
            keys = select_lowdim_obs_keys(list(obs_grp.keys()))
            dim = sum(int(np.asarray(obs_grp[k]).reshape(np.asarray(obs_grp[k]).shape[0], -1).shape[1]) for k in keys)
            print(f"  low-dim obs keys: {keys}")
            return dim
    except Exception as e:  # noqa: BLE001
        print(f"  (could not probe obs dim: {e})")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="datasets/libero/libero_goal_goal_embs.npz",
        help="Output .npz cache path.",
    )
    ap.add_argument("--encoder", choices=list(ENCODERS), default="minilm")
    args = ap.parse_args()

    infos = get_task_infos()
    names = [t["name"] for t in infos]
    instructions = [t["language"] for t in infos]

    print(f"LIBERO-Goal: {len(infos)} tasks")
    for t in infos:
        print(f"  [{t['index']}] {t['name']}: {t['language']!r}")

    print(f"\nEncoding with '{args.encoder}'...")
    embeddings = ENCODERS[args.encoder](instructions)
    emb_dim = int(embeddings.shape[1])

    save_goal_embeddings(args.out, names, embeddings, instructions)
    print(f"\nSaved {embeddings.shape} embeddings -> {args.out}")
    print(f"goal_emb_dim = {emb_dim}")

    obs_dim = probe_obs_dim(infos[0]["demo_file"])
    if obs_dim is not None:
        print(
            f"\n>>> Set config/config.json environments.libero_goal.state_dim = "
            f"{obs_dim} + {emb_dim} = {obs_dim + emb_dim}  (frame_stack=1)"
        )


if __name__ == "__main__":
    main()
