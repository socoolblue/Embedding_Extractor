"""
MACE-MPA-0 (medium) embedding extractor.
CIF directory -> 공식 get_descriptors(invariants_only=True) -> (N_atoms, 256) -> mean pooling.
후킹 불필요: MACE는 디스크립터 추출 API를 공식 제공.

Install (별도 환경 권장):
    pip install mace-torch ase numpy
Usage:
    python extract_mace.py --cif_dir ./cifs --out mace_emb.npz [--device cuda]
"""
import argparse, glob, os
import numpy as np
import torch
from ase.io import read
try:
    from tqdm import tqdm
except ImportError:  # tqdm 미설치 시 단순 진행 출력으로 대체
    def tqdm(it, **kw):
        total = kw.get("total", None) or (len(it) if hasattr(it, "__len__") else None)
        def gen():
            for i, x in enumerate(it, 1):
                if i % 100 == 0 or i == total:
                    print(f"  progress: {i}/{total}", flush=True)
                yield x
        return gen()
from mace.calculators import mace_mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out", default="mace_emb.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1) fully-trained checkpoint (auto-download on first run)
    #    model="medium-mpa-0" -> MACE-MPA-0 medium (MPtrj + sAlex, MIT-compatible)
    calc = mace_mp(model="medium-mpa-0", device=args.device, default_dtype="float64")

    # 2) CIF 순회 — invariant(l=0) 채널만: nchannels(128) x nlayers(2) = 256
    cif_paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    if not cif_paths:
        raise SystemExit(f"[error] no .cif files found in: {os.path.abspath(args.cif_dir)}")
    ids, embs = [], []
    for p in tqdm(cif_paths, desc="extracting", unit="cif"):
        atoms = read(p)
        desc = calc.get_descriptors(atoms, invariants_only=True)  # (N_atoms, 256)
        desc = np.asarray(desc)
        embs.append(desc.mean(axis=0).astype(np.float32))         # mean pooling
        ids.append(os.path.splitext(os.path.basename(p))[0])
        if len(ids) == 1:
            print(f"[info] detected descriptor dim: {desc.shape[1]}")

    embs = np.vstack(embs)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=embs)
    print(f"[done] {embs.shape} -> {args.out}  (dim={embs.shape[1]})")

if __name__ == "__main__":
    main()
