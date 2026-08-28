"""
DPA-3.1-3M embedding extractor (DeePMD-kit v3, PyTorch backend).
공식 DeepPot.eval_descriptor() API 사용 — descriptor net 출력 = fitting net(readout) 직전 표현.
CIF directory -> per-atom descriptor -> mean pooling -> npz.

Install (별도 환경 권장):
    pip install deepmd-kit[torch] ase pymatgen numpy
Model:
    DPA-3.1-3M 체크포인트(.pth)를 AIS Square 또는 HuggingFace(deepmodeling)에서 수동 다운로드.
Usage:
    python extract_dpa.py --cif_dir ..\\cifs --model DPA-3.1-3M.pth --out dpa_emb.npz
    (multitask 체크포인트인 경우 --head 지정. 미지정 시 에러 메시지가 head 목록을 보여줌
     -> MPtrj 계열 head를 선택할 것. 예: --head MP_traj)
"""
import argparse, glob, os
import numpy as np
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--model", required=True, help="path to DPA-3.1-3M checkpoint (.pth/.pt)")
    ap.add_argument("--out", default="dpa_emb.npz")
    ap.add_argument("--head", default=None,
                    help="multitask head name (MPtrj 정합 head 권장, 예: MP_traj)")
    args = ap.parse_args()

    from deepmd.infer.deep_pot import DeepPot
    kwargs = {"head": args.head} if args.head else {}
    dp = DeepPot(args.model, **kwargs)

    # 원소기호 -> 모델 type_map 인덱스 (원자번호 아님!)
    type_map = dp.get_type_map()
    sym2idx = {s: i for i, s in enumerate(type_map)}
    print(f"[info] model type_map ({len(type_map)} elements): {type_map[:10]} ...")

    cif_paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    if not cif_paths:
        raise SystemExit(f"[error] no .cif files found in: {os.path.abspath(args.cif_dir)}")

    ids, embs = [], []
    for p in tqdm(cif_paths, desc="extracting", unit="cif"):
        atoms = read(p)
        symbols = atoms.get_chemical_symbols()
        unknown = sorted(set(symbols) - set(sym2idx))
        if unknown:
            print(f"[skip] {os.path.basename(p)}: elements not in type_map: {unknown}")
            continue

        coords = atoms.get_positions().reshape(1, -1)          # (1, natoms*3)
        cells = atoms.get_cell().array.reshape(1, 9)           # (1, 9), PBC
        atype = [sym2idx[s] for s in symbols]                  # type indices

        desc = dp.eval_descriptor(coords, cells, atype)        # (1, natoms, D)
        desc = np.asarray(desc).reshape(len(symbols), -1)      # (natoms, D)
        embs.append(desc.mean(axis=0).astype(np.float32))      # mean pooling
        ids.append(os.path.splitext(os.path.basename(p))[0])
        if len(ids) == 1:
            print(f"[info] detected descriptor dim: {desc.shape[1]}")

    if not embs:
        raise SystemExit("[error] no structure was processed successfully")
    embs = np.vstack(embs)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=embs)
    print(f"[done] {embs.shape} -> {args.out}  (dim={embs.shape[1]})")

if __name__ == "__main__":
    main()
