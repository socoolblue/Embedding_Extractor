"""
MatterSim-v1.0.0-5M embedding extractor.
CIF directory -> per-structure mean-pooled node features (readout 직전 atom_attr).

Install (별도 환경 권장):
    pip install mattersim ase numpy
Usage:
    python extract_mattersim.py --cif_dir ./cifs --out mattersim_emb.npz [--device cuda]
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
from mattersim.forcefield.potential import Potential, batch_to_dict
from mattersim.datasets.utils.build import build_dataloader

CAPTURED = {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out", default="mattersim_emb.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1) fully-trained checkpoint (auto-download on first run)
    potential = Potential.from_checkpoint(
        load_path="mattersim-v1.0.0-5M.pth",
        device=args.device,
        load_training_state=False,   # 추론 전용: optimizer 상태 불필요
    )
    model = potential.model  # M3Gnet
    model.eval()
    dim = model.model_args.get("units", None) if hasattr(model, "model_args") else None
    print(f"[info] hidden units reported by checkpoint: {dim}")

    # 2) hook: energy head(self.final)에 들어가기 직전의 atom_attr 캡처
    def pre_hook(module, inputs):
        CAPTURED["atom_attr"] = inputs[0].detach().cpu()
    handle = model.final.register_forward_pre_hook(pre_hook)

    # 3) CIF 순회 (구조당 1개씩: 후킹-배치 매핑의 모호성 제거)
    cif_paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    if not cif_paths:
        raise SystemExit(f"[error] no .cif files found in: {os.path.abspath(args.cif_dir)}")
    ids, embs = [], []
    with torch.no_grad():
        for p in tqdm(cif_paths, desc="extracting", unit="cif"):
            atoms = read(p)
            dl = build_dataloader([atoms], only_inference=True, batch_size=1)
            for batch in dl:
                inp = batch_to_dict(batch, device=args.device)  # 공식 변환 경로
                model(inp)
            feat = CAPTURED.pop("atom_attr")            # (N_atoms, units)
            embs.append(feat.mean(dim=0).numpy().astype(np.float32))
            ids.append(os.path.splitext(os.path.basename(p))[0])
            if len(ids) == 1:
                print(f"[info] detected embedding dim: {feat.shape[1]}")
    handle.remove()

    embs = np.vstack(embs)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=embs)
    print(f"[done] {embs.shape} -> {args.out}  (dim={embs.shape[1]})")

if __name__ == "__main__":
    main()
