"""
SevenNet-MF-ompa embedding extractor.
CIF directory -> readout 직전 node feature data['x'] (최종층 128x0e, 순수 스칼라) mean pooling.

Install (별도 환경 권장):
    pip install sevenn ase numpy
Usage:
    python extract_sevennet.py --cif_dir ./cifs --out sevennet_emb.npz [--device cuda]
주의: multi-fidelity 모델이므로 modal='mpa'(MP PBE+U 정합)로 고정. 논문에 명시할 것.
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
from sevenn.calculator import SevenNetCalculator

CAPTURED = {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out", default="sevennet_emb.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--modal", default="mpa", choices=["mpa", "omat24"])
    args = ap.parse_args()

    # 1) fully-trained checkpoint (auto-download on first run)
    calc = SevenNetCalculator(model="7net-mf-ompa", modal=args.modal, device=args.device)
    model = calc.model
    model.eval()

    # 2) readout 첫 모듈을 이름으로 탐색 -> 그 입력 dict의 data['x']가
    #    마지막 conv 층 출력(최종 node feature, 128x0e)
    readout_names = ("reduce_input_to_hidden", "readout_FCN")
    target = None
    for name, module in model.named_modules():
        if name.split(".")[-1] in readout_names:
            target = module
            break
    assert target is not None, "readout module not found; check sevenn version"

    def pre_hook(module, inputs):
        data = inputs[0]
        CAPTURED["x"] = data["x"].detach().cpu()
    handle = target.register_forward_pre_hook(pre_hook)

    # 3) CIF 순회 — calculator 경로 재사용(그래프 전처리 일관성 보장)
    cif_paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    if not cif_paths:
        raise SystemExit(f"[error] no .cif files found in: {os.path.abspath(args.cif_dir)}")
    ids, embs = [], []
    for p in tqdm(cif_paths, desc="extracting", unit="cif"):
        atoms = read(p)
        atoms.calc = calc
        atoms.get_potential_energy()                 # forward 1회 (힘=autograd라 no_grad 금지)
        feat = CAPTURED.pop("x")                     # (N_atoms, 128) scalar-only
        embs.append(feat.mean(dim=0).numpy().astype(np.float32))
        ids.append(os.path.splitext(os.path.basename(p))[0])
        if len(ids) == 1:
            print(f"[info] detected embedding dim: {feat.shape[1]}")
    handle.remove()

    embs = np.vstack(embs)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=embs)
    print(f"[done] {embs.shape} -> {args.out}  (dim={embs.shape[1]}, modal={args.modal})")

if __name__ == "__main__":
    main()
