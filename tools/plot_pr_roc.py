
# tools/plot_pr_roc.py
import argparse, os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
import csv, math

def load_run(arg):
    name, path = arg.split("=",1)
    data = np.load(path)
    y = data["y"] if "y" in data else data["labels"]
    s = data["score"] if "score" in data else data["p"] if "p" in data else data["s"]
    y = y.astype(np.float32).ravel(); s = s.astype(np.float32).ravel()
    assert y.shape==s.shape
    return name, y, s

def bootstrap_ci(func, y, s, n=10000, seed=0):
    rng = np.random.default_rng(seed); idx=np.arange(len(y))
    vals=[]
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        try: vals.append(func(y[b], s[b]))
        except: pass
    a=np.array(vals); 
    return float(a.mean()), float(np.percentile(a,2.5)), float(np.percentile(a,97.5))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="out_figs")
    ap.add_argument("--bootstrap", type=int, default=0)
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs=[load_run(r) for r in args.run]

    # PR
    plt.figure(figsize=(6,5))
    for n,y,s in runs:
        prec, rec, _ = precision_recall_curve(y,s)
        apv = average_precision_score(y,s)
        plt.plot(rec, prec, label=f"{n} (AP={apv:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"pr_curve.png"), dpi=160); plt.close()

    # ROC
    plt.figure(figsize=(6,5))
    for n,y,s in runs:
        fpr, tpr, _ = roc_curve(y,s); auc = roc_auc_score(y,s)
        plt.plot(fpr, tpr, label=f"{n} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"roc_curve.png"), dpi=160); plt.close()

    # CSV
    rows=[("Method","N","Pos","Neg","AUROC","AUPRC","AUROC mean(95% CI)","AUPRC mean(95% CI)")]
    for n,y,s in runs:
        auroc = roc_auc_score(y,s); auprc = average_precision_score(y,s)
        if args.bootstrap>0:
            m1,l1,h1 = bootstrap_ci(roc_auc_score,y,s,args.bootstrap,0)
            m2,l2,h2 = bootstrap_ci(average_precision_score,y,s,args.bootstrap,1)
            ci1=f"{m1:.3f} ({l1:.3f},{h1:.3f})"; ci2=f"{m2:.3f} ({l2:.3f},{h2:.3f})"
        else:
            ci1=""; ci2=""
        rows.append((n,len(y),int(y.sum()),int(len(y)-y.sum()),f"{auroc:.4f}",f"{auprc:.4f}",ci1,ci2))
    with open(os.path.join(args.out,"summary.csv"),"w",newline="") as f:
        import csv; w=csv.writer(f); w.writerows(rows)
    print("[OK] Saved PR/ROC and summary.csv")

if __name__=="__main__":
    main()
