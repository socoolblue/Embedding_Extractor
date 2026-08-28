import argparse, os, glob
import numpy as np
import torch
from pymatgen.core import Structure
from chgnet.model import CHGNet
from tqdm import tqdm

# ---------------- 설정 ----------------
ap = argparse.ArgumentParser(description="CHGNet graph-level embedding extractor")
ap.add_argument("--cif_dir", required=True, help="CIF 파일이 들어있는 디렉터리")
ap.add_argument("--out", default="chgnet_emb.npz", help="출력 npz 경로")
args = ap.parse_args()

out_dir = os.path.dirname(os.path.abspath(args.out))
os.makedirs(out_dir, exist_ok=True)

# ---------------- 1) 모델 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CHGNet.load().to(device)
model.eval()
print(f"device: {device} | CHGNet loaded")

# ---------------- 2) hook (mlp 입력 = graph-level 64D) ----------------
buf = {}
def pre_hook_fn(module, inputs):
    arr = inputs[0].detach().cpu().numpy()
    if arr.ndim == 2:
        buf["vec"] = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    elif arr.ndim == 1:
        buf["vec"] = arr
    else:
        raise RuntimeError(f"Unexpected mlp input shape: {arr.shape}")

handle = model.mlp.register_forward_pre_hook(pre_hook_fn)

# ---------------- 3) CIF 목록 ----------------
cif_paths = sorted(glob.glob(os.path.join(args.cif_dir, "*.cif")))
print(f"found {len(cif_paths)} cif files")
if not cif_paths:
    raise SystemExit(f"[error] no .cif files found in: {os.path.abspath(args.cif_dir)}")

# ---------------- 4) 추출 ----------------
ids, embs, failed = [], [], []
for p in tqdm(cif_paths, desc="CHGNet", unit="cif"):
    buf.clear()
    try:
        s = Structure.from_file(p)
        with torch.no_grad():
            g = model.graph_converter(s).to(device)
            _ = model([g])
        if "vec" not in buf:
            raise RuntimeError("hook did not fire")
        embs.append(buf["vec"].astype(np.float32))
        ids.append(os.path.splitext(os.path.basename(p))[0])
        if len(ids) == 1:
            print(f"[info] embedding dim: {len(buf['vec'])}")
    except Exception as e:
        failed.append((os.path.basename(p), str(e)[:80]))
        continue

handle.remove()

# ---------------- 5) 저장 ----------------
if not embs:
    print(f"[error] 모든 파일 실패 ({len(cif_paths)}개)")
    for f, e in failed[:10]:
        print(f"  [fail] {f}: {e}")
    raise SystemExit(1)

embs = np.vstack(embs)
ids  = np.array(ids)
assert len(ids) == len(embs)
np.savez_compressed(args.out, ids=ids, embeddings=embs)

print(f"\n[done] {embs.shape} -> {args.out}")
print(f"성공 {len(ids)} / 시도 {len(cif_paths)} | 실패 {len(failed)}")
for f, e in failed[:10]:
    print(f"  [fail] {f}: {e}")
if len(failed) > 10:
    print(f"  ... 외 {len(failed)-10}개")