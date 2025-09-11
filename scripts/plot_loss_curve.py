import os, json, glob
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "H"]

def ema(series, alpha=0.6):
    if not series: return series
    out, prev = [], series[0]
    for x in series:
        prev = alpha * x + (1 - alpha) * prev
        out.append(prev)
    return out

def load_history(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        h = json.load(f)
    parsed = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "train_f1":[], "val_f1":[]}
    if "train" in h and isinstance(h["train"], list):
        parsed["train_loss"] = [x.get("loss") for x in h["train"] if x.get("loss") is not None]
        parsed["val_loss"]   = [x.get("loss") for x in h.get("val", []) if x.get("loss") is not None]
        if any("acc" in x for x in h["train"]): parsed["train_acc"] = [x.get("acc") for x in h["train"] if x.get("acc") is not None]
        if any("acc" in x for x in h.get("val", [])): parsed["val_acc"] = [x.get("acc") for x in h.get("val", []) if x.get("acc") is not None]
        if any("f1" in x for x in h["train"]): parsed["train_f1"] = [x.get("f1") for x in h["train"] if x.get("f1") is not None]
        if any("f1" in x for x in h.get("val", [])): parsed["val_f1"] = [x.get("f1") for x in h.get("val", []) if x.get("f1") is not None]
    else:
        for k in parsed.keys():
            if isinstance(h.get(k), list):
                parsed[k] = [v for v in h[k] if v is not None]
    return parsed

def mark_best_point(ax, y_vals, mode="min", color=None, marker="o"):
    if not y_vals:
        return None
    idx = (min if mode=="min" else max)(range(len(y_vals)), key=lambda i: y_vals[i])
    val = y_vals[idx]
    ax.scatter([idx+1], [val],
               s=70, marker=marker,
               c=color if color is not None else "red",
               edgecolors="black", linewidths=1.0, zorder=5)
    return idx+1, val

def plot_group(histories, key, title, ylabel, out_path, smooth_alpha=None, mark_best=False):
    fig, ax = plt.subplots(figsize=(11,6))
    any_curve = False

    # 为了获得每条曲线的颜色，先保存 line 对象
    for i, (name, h) in enumerate(histories.items()):
        vals = h.get(key, [])
        if not vals:
            continue
        if smooth_alpha:
            vals = ema(vals, smooth_alpha)
        # 画线，取回颜色
        line = ax.plot(range(1, len(vals)+1), vals, label=name)
        color = line[0].get_color()
        if mark_best:
            mode = "min" if "loss" in key else "max"
            marker = MARKERS[i % len(MARKERS)]
            mark_best_point(ax, vals, mode=mode, color=color, marker=marker)
        any_curve = True

    if not any_curve:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist_dir", type=str, default="outputs")
    ap.add_argument("--pattern", type=str, default="history_*.json")
    ap.add_argument("--out_dir", type=str, default="figs")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA alpha (0~1], 0=disable")
    ap.add_argument("--mark_best", action="store_true", help="mark best epoch with distinct markers")
    args = ap.parse_args()

    hist_dir = Path(args.hist_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(hist_dir / args.pattern)))
    if not files:
        print(f"[WARN] No history files in {hist_dir} matching {args.pattern}")
        return

    histories = {}
    for fp in files:
        name = Path(fp).stem.replace("history_", "")
        try:
            histories[name] = load_history(fp)
        except Exception as e:
            print(f"[WARN] Failed to parse {fp}: {e}")

    alpha = args.ema if args.ema and args.ema > 0 else None

    plot_group(histories, "train_loss", "Training Loss", "Loss",
               out_dir / "train_loss.png", smooth_alpha=alpha, mark_best=args.mark_best)
    plot_group(histories, "val_loss",   "Validation Loss", "Loss",
               out_dir / "val_loss.png",   smooth_alpha=alpha, mark_best=args.mark_best)
    plot_group(histories, "train_acc",  "Training Accuracy", "Accuracy",
               out_dir / "train_acc.png",  smooth_alpha=alpha, mark_best=args.mark_best)
    plot_group(histories, "val_acc",    "Validation Accuracy", "Accuracy",
               out_dir / "val_acc.png",    smooth_alpha=alpha, mark_best=args.mark_best)
    plot_group(histories, "train_f1",   "Training F1", "F1",
               out_dir / "train_f1.png",   smooth_alpha=alpha, mark_best=args.mark_best)
    plot_group(histories, "val_f1",     "Validation F1", "F1",
               out_dir / "val_f1.png",     smooth_alpha=alpha, mark_best=args.mark_best)

if __name__ == "__main__":
    main()
