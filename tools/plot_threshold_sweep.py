# tools/plot_threshold_sweep_multi.py
# Compare multiple models' threshold sweeps on validation scores.
# Each --run accepts either:
#   --run "Label=path/to/scores_val.npz"
# or
#   --run path/to/scores_val.npz      (label will be inferred)
#
# Examples:
#   python tools/plot_threshold_sweep.py
#     --run "UNet-CAM=ckpts/unet_cam/scores_val.npz"
#     --run "UNet-MP=ckpts/unet_mp/scores_val.npz"
#     --run "UNet-Pseudo=ckpts/unet_pseudo/scores_val.npz"
#     --run "ResNet18=ckpts/cls_resnet18/scores_val.npz"
#     --run "EffNetB0=ckpts/cls_effb0/scores_val.npz"
#     --metric bac --out out_figs --bootstrap 0
#
import argparse, os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

def parse_run_arg(s: str):
    if "=" in s:
        name, path = s.split("=", 1)
        return name.strip(), path.strip()
    path = s.strip()
    p = Path(path)
    # infer label from parent dir (e.g., ckpts/unet_cam) or file stem
    label = p.parent.name if p.parent.name else p.stem
    return label, path

def metric_at_threshold(y, s, t, metric="bac"):
    yhat = (s >= t).astype(int)
    if metric == "f1":
        return f1_score(y, yhat, zero_division=0)
    elif metric == "bac":
        return balanced_accuracy_score(y, yhat)
    elif metric == "youden":
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
        tpr = tp / (tp + fn + 1e-12); tnr = tn / (tn + fp + 1e-12)
        return tpr + tnr - 1.0
    else:
        raise ValueError("metric must be 'bac' | 'f1' | 'youden'")

def sweep_curve(y, s, grid, metric="bac"):
    vals = np.array([metric_at_threshold(y, s, t, metric) for t in grid], dtype=float)
    i = int(np.argmax(vals))
    return vals, float(grid[i]), float(vals[i])

def bootstrap_best(y, s, grid, metric="bac", B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    best_t = np.zeros(B, dtype=float)
    best_m = np.zeros(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]; ss = s[idx]
        vals, t_star, m_star = sweep_curve(yy, ss, grid, metric)
        best_t[b] = t_star; best_m[b] = m_star
    lo_t, hi_t = np.percentile(best_t, [2.5, 97.5])
    lo_m, hi_m = np.percentile(best_m, [2.5, 97.5])
    return (best_t.mean(), lo_t, hi_t), (best_m.mean(), lo_m, hi_m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help='Repeat. Format "Label=path/to/scores_val.npz" or just the path.')
    ap.add_argument("--metric", default="bac", choices=["bac","f1","youden"])
    ap.add_argument("--out", default="out_figs")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--grid", type=int, default=1001, help="number of thresholds between 0 and 1")
    ap.add_argument("--bootstrap", type=int, default=0, help="B resamples; 0 = disable")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    grid = np.linspace(0, 1, args.grid)

    summary = []
    plt.figure(figsize=(7,6))

    for r in args.run:
        label, path = parse_run_arg(r)
        data = np.load(path)
        y = data["y"].astype(int).ravel()
        s = data["score"].astype(float).ravel()

        vals, t_star, m_star = sweep_curve(y, s, grid, args.metric)

        # optional bootstrap for CI of best threshold/metric
        ci_t, ci_m = (None, None)
        if args.bootstrap and args.bootstrap > 0:
            ci_t, ci_m = bootstrap_best(y, s, grid, args.metric, args.bootstrap, args.seed)
            label_plot = f"{label} (best {args.metric.upper()}={m_star:.3f} [{ci_m[1]:.3f},{ci_m[2]:.3f}], t*={t_star:.3f} [{ci_t[1]:.3f},{ci_t[2]:.3f}])"
        else:
            label_plot = f"{label} (best {args.metric.upper()}={m_star:.3f}, t*={t_star:.3f})"

        plt.plot(grid, vals, label=label_plot)
        plt.scatter([t_star], [m_star], s=24)

        row = {
            "label": label,
            "metric": args.metric,
            "best_value": m_star,
            "best_threshold": t_star,
        }
        if ci_t and ci_m:
            row.update({
                "best_value_ci_lo": ci_m[1], "best_value_ci_hi": ci_m[2],
                "best_threshold_ci_lo": ci_t[1], "best_threshold_ci_hi": ci_t[2],
            })
        summary.append(row)

    plt.xlabel("Threshold"); plt.ylabel(args.metric.upper())
    plt.title(f"{args.metric.upper()} vs Threshold (validation)")
    if args.ylim is not None:
        plt.ylim(args.ylim[0], args.ylim[1])
    plt.legend(fontsize=9)
    out_png = os.path.join(args.out, f"threshold_sweep_multi_{args.metric}.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=args.dpi); plt.close()

    # write CSV + JSON summary
    import csv
    out_csv = os.path.join(args.out, f"threshold_sweep_multi_{args.metric}.csv")
    keys = sorted({k for row in summary for k in row.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(summary)
    out_json = os.path.join(args.out, f"threshold_sweep_multi_{args.metric}.json")
    with open(out_json, "w") as f:
        f.write(json.dumps(summary, indent=2))
    print("[OK]", out_png)
    print("[OK]", out_csv)
    print("[OK]", out_json)

if __name__ == "__main__":
    main()

