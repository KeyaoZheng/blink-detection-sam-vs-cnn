
# tools/plot_training_curves.py
import argparse, os, json
import matplotlib.pyplot as plt

def load_log(run_dir):
    log_path = os.path.join(run_dir, "train_log.json")
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"train_log.json not found in {run_dir}")
    with open(log_path, "r") as f:
        log = json.load(f)
    epochs = [d.get("epoch") for d in log]
    train_loss = [d.get("train_loss", None) for d in log]
    auroc = [d.get("AUROC", None) for d in log]
    auprc = [d.get("AUPRC", None) for d in log]
    name = os.path.basename(run_dir.rstrip('/\\'))
    return {"name": name, "epochs": epochs, "train_loss": train_loss, "AUROC": auroc, "AUPRC": auprc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="Repeat: path to one run dir containing train_log.json")
    ap.add_argument("--metric", default="AUPRC", choices=["AUPRC","AUROC"], help="Which validation metric to plot")
    ap.add_argument("--out", default="out_figs", help="Output directory for figures")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logs = [load_log(d) for d in args.run]

    # Train Loss vs Epoch
    plt.figure(figsize=(6,5))
    for L in logs:
        ep = [e for e in L["epochs"] if e is not None]
        tr = [t for t in L["train_loss"] if t is not None]
        if len(ep) == len(tr) and len(ep) > 0:
            plt.plot(ep, tr, label=L["name"])
    plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Train Loss vs Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "train_loss_vs_epoch.png"), dpi=args.dpi); plt.close()

    # Val Metric vs Epoch
    plt.figure(figsize=(6,5))
    key = args.metric
    for L in logs:
        ep = [e for e in L["epochs"] if e is not None]
        mv = [m for m in L[key] if m is not None]
        if len(ep) == len(mv) and len(ep) > 0:
            plt.plot(ep, mv, label=f"{L['name']} ({key})")
    plt.xlabel("Epoch"); plt.ylabel(key); plt.title(f"{key} vs Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.out, f"{key}_vs_epoch.png"), dpi=args.dpi); plt.close()

    print("[OK] Saved curves to", args.out)

if __name__ == "__main__":
    main()
