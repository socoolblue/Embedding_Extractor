"""
GRACE-2L-OAM embedding extractor (TensorFlow / tensorpotential).
공식 ExtractBasisFunctions API 사용 — 후킹/몽키패치 불필요.
CIF directory -> readout 직전 per-atom invariant basis -> mean pooling -> npz.

Install (별도 환경 권장; TF 기반이므로 PyTorch 모델들과 분리):
    pip install tensorpotential tensorflow ase numpy
Usage:
    1) 체크포인트 1회 다운로드:  grace_models download GRACE-2L-OAM
       (기본 위치: ~/.cache/grace/checkpoints/GRACE-2L-OAM/)
    2) python extract_grace.py --cif_dir ./cifs --out grace_emb.npz
저장 내용: 최종층(2L_basis, primary) + 1L/2L concat(보조) 두 버전 모두.
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

from tensorpotential.tensorpot import TensorPotential
from tensorpotential.calculator import TPCalculator
from tensorpotential.tpmodel import ExtractBasisFunctions
from tensorpotential.instructions.base import load_instructions

DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/grace/checkpoints/GRACE-2L-OAM")

def read_param_dtype(model_dir: str):
    """model.yaml에 명시된 param_dtype을 읽는다. 없으면 float64(OAM 체크포인트 기본)."""
    import tensorflow as tf
    yaml_path = os.path.join(model_dir, "model.yaml")
    try:
        with open(yaml_path) as f:
            for line in f:
                if "param_dtype" in line:
                    if "float64" in line or "double" in line:
                        return tf.float64
                    if "float32" in line:
                        return tf.float32
    except OSError:
        pass
    return tf.float64  # GRACE-2L-OAM 체크포인트는 float64로 저장됨

def build_basis_calculator(model_dir: str) -> TPCalculator:
    """공식 FAQ 레시피 + param_dtype 정합: instruction graph 로드 -> ExtractBasisFunctions 장착."""
    dtype = read_param_dtype(model_dir)
    print(f"[info] building model with param_dtype={dtype.name}")
    instr = load_instructions(os.path.join(model_dir, "model.yaml"))
    tp = TensorPotential(
        instr,
        model_compute_function=ExtractBasisFunctions(
            extract_1L_basis=True,
            extract_2L_basis=False,  # OAM 체크포인트는 1L+2L basis를 단일 I_out으로 통합
        ),
        param_dtype=dtype,           # 체크포인트 저장 정밀도와 일치 필수
    )
    tp.load_checkpoint(
        checkpoint_name=os.path.join(model_dir, "checkpoint"), verbose=True
    )
    tp.model.decorate_compute_function(jit_compile=True)
    return TPCalculator(
        model=tp.model,
        truncate_extras_by_natoms=True,          # padding 원자 자동 제거
        extra_properties=["1L_basis"],           # = 통합 I_out (양 층 invariant 전부)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", required=True)
    ap.add_argument("--out", default="grace_emb.npz")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    args = ap.parse_args()

    calc = build_basis_calculator(args.model_dir)

    cif_paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
    if not cif_paths:
        raise SystemExit(f"[error] no .cif files found in: {os.path.abspath(args.cif_dir)}")
    ids, embs = [], []
    for p in tqdm(cif_paths, desc="extracting", unit="cif"):
        atoms = read(p)
        atoms.calc = calc
        atoms.get_potential_energy()             # forward 1회 트리거
        basis = np.asarray(calc.results["1L_basis"])  # (N_atoms, d) 통합 I_out
        embs.append(basis.mean(axis=0).astype(np.float32))  # mean pooling + 용량 절감
        ids.append(os.path.splitext(os.path.basename(p))[0])
        if len(ids) == 1:
            print(f"[info] detected embedding dim (unified I_out): {basis.shape[1]}")

    embs = np.vstack(embs)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=embs)
    print(f"[done] {embs.shape} -> {args.out}  (dim={embs.shape[1]})")

if __name__ == "__main__":
    main()
