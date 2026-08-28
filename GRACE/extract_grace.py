import argparse, glob, os
import numpy as np
from ase.io import read
try:
    from tqdm import tqdm
except ImportError:  # fall back to simple progress printing if tqdm is not installed
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
    """Read the param_dtype declared in model.yaml. Falls back to float64 (the OAM checkpoint default)."""
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
    return tf.float64  # the GRACE-2L-OAM checkpoint is stored in float64

def build_basis_calculator(model_dir: str) -> TPCalculator:
    """Official FAQ recipe + param_dtype match: load the instruction graph -> attach ExtractBasisFunctions."""
    dtype = read_param_dtype(model_dir)
    print(f"[info] building model with param_dtype={dtype.name}")
    instr = load_instructions(os.path.join(model_dir, "model.yaml"))
    tp = TensorPotential(
        instr,
        model_compute_function=ExtractBasisFunctions(
            extract_1L_basis=True,
            extract_2L_basis=False,  # the OAM checkpoint merges the 1L+2L basis into a single I_out
        ),
        param_dtype=dtype,           # must match the precision the checkpoint was saved in
    )
    tp.load_checkpoint(
        checkpoint_name=os.path.join(model_dir, "checkpoint"), verbose=True
    )
    tp.model.decorate_compute_function(jit_compile=True)
    return TPCalculator(
        model=tp.model,
        truncate_extras_by_natoms=True,          # automatically drop padding atoms
        extra_properties=["1L_basis"],           # = the merged I_out (all invariants from both layers)
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
        atoms.get_potential_energy()             # trigger a single forward pass
        basis = np.asarray(calc.results["1L_basis"])  # (N_atoms, d) merged I_out
        embs.append(basis.mean(axis=0).astype(np.float32))  # mean pooling + smaller output size
        ids.append(os.path.splitext(os.path.basename(p))[0])
        if len(ids) == 1:
            print(f"[info] detected embedding dim (unified I_out): {basis.shape[1]}")

    embs = np.vstack(embs)
    np.savez_compressed(args.out, ids=np.array(ids), embeddings=embs)
    print(f"[done] {embs.shape} -> {args.out}  (dim={embs.shape[1]})")

if __name__ == "__main__":
    main()