import argparse, os, shutil, random
from pathlib import Path

def collect(src):
    src = Path(src)
    cls = {
        "blink":  [src/"closedLeftEyes", src/"closedRightEyes"],
        "nonblink":[src/"openLeftEyes",   src/"openRightEyes"  ],
    }
    files = {"blink":[], "nonblink":[]}
    for k, dirs in cls.items():
        for d in dirs:
            if not d.exists(): raise FileNotFoundError(d)
            for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
                files[k] += list(d.rglob(ext))
    return files

def split_and_copy(files, dst, seed=42, ratios=(0.8,0.1,0.1)):
    random.seed(seed)
    dst = Path(dst)
    for split in ("train","val","test"):
        (dst/split/"blink").mkdir(parents=True, exist_ok=True)
        (dst/split/"nonblink").mkdir(parents=True, exist_ok=True)

    for cls, paths in files.items():
        paths = list(paths); random.shuffle(paths)
        n = len(paths); n_tr = int(n*ratios[0]); n_va = int(n*ratios[1])
        parts = {"train": paths[:n_tr],
                 "val":   paths[n_tr:n_tr+n_va],
                 "test":  paths[n_tr+n_va:]}
        for split, lst in parts.items():
            for p in lst:
                out = dst/split/cls/p.name
                shutil.copy2(p, out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help=".../train_datasets")
    ap.add_argument("--dst", required=True, help="output root to create train/val/test/blink/nonblink")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = collect(args.src)
    print({k:len(v) for k,v in files.items()})
    split_and_copy(files, args.dst, seed=args.seed)
    print("Done. Prepared at:", args.dst)


