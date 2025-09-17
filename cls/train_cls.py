
# cls/train_cls.py
# Binary eye open/closed classifier training: ResNet-18 or EfficientNet-B0
# Example:
#   python -m cls.train_cls --data_root data_splits --arch resnet18 --epochs 20 --bs 64 --lr 3e-4 --img_size 128 256 --out ckpts/cls_resnet18
#   python -m cls.train_cls --data_root data_splits --arch efficientnet_b0 --pretrained 1 --epochs 20 --bs 64 --lr 2e-4 --img_size 128 256 --out ckpts/cls_effb0
import argparse, os, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

def build_model(arch: str, pretrained: bool):
    arch = arch.lower()
    if arch == "resnet18":
        from torchvision import models
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m
    elif arch in ("efficientnet_b0","effnet_b0","efficientnet-b0","effnet"):
        try:
            import timm
        except Exception as e:
            raise RuntimeError("Please `pip install timm` for EfficientNet-B0") from e
        m = timm.create_model("efficientnet_b0", num_classes=1, pretrained=pretrained)
        return m
    else:
        raise ValueError(f"Unknown arch: {arch}")

class EyeClsDataset(Dataset):
    def __init__(self, root, split, size_hw, norm: bool):
        self.items = []
        for cls in ["open","closed"]:
            d = Path(root)/split/cls
            if not d.exists(): continue
            for p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png")) + list(d.glob("*.bmp"))):
                y = 1.0 if cls=="open" else 0.0
                self.items.append((p, y))
        H, W = size_hw
        tfs = [transforms.Resize((H, W)), transforms.ToTensor()]
        if norm:
            tfs.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
        self.tf = transforms.Compose(tfs)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, y = self.items[idx]
        im = Image.open(p).convert("RGB")
        x = self.tf(im)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y, str(p)

def eval_metrics(model, loader, device):
    model.eval(); ys=[]; ps=[]
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); y = y.squeeze(1).to(device)
            logits = model(x).squeeze(1)
            prob = torch.sigmoid(logits)
            ys.append(y.detach().cpu().numpy()); ps.append(prob.detach().cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return {
        "AUROC": float(roc_auc_score(ys, ps)),
        "AUPRC": float(average_precision_score(ys, ps))
    }, ys, ps

def find_threshold(y, s, how='bac'):
  
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s).astype(float).ravel()
    grid = np.linspace(0, 1, 1001)
    best_t, best = 0.5, -1.0
    for t in grid:
        yhat = (s >= t).astype(int)
        if how == 'f1':
            m = f1_score(y, yhat, zero_division=0)
        elif how == 'bac':
            m = balanced_accuracy_score(y, yhat)
        elif how == 'youden':
            # Youden's J = TPR + TNR - 1
            tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
            tpr = tp / (tp + fn + 1e-12)
            tnr = tn / (tn + fp + 1e-12)
            m = tpr + tnr - 1.0
        else:
            raise ValueError("how must be 'bac' | 'f1' | 'youden'")
        if m > best:
            best, best_t = m, float(t)
    return best_t

def eval_with_threshold(y, s, t):
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s).astype(float).ravel()
    yhat = (s >= t).astype(int)
    return {
        "Accuracy":      float(accuracy_score(y, yhat)),
        "F1":            float(f1_score(y, yhat, zero_division=0)),
        "BalancedAcc":   float(balanced_accuracy_score(y, yhat)),
        "threshold":     float(t),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = bool(args.pretrained)  # if using pretrained weights, use ImageNet norm
    train_ds = EyeClsDataset(args.data_root, "train", (args.img_h,args.img_w), norm)
    val_ds   = EyeClsDataset(args.data_root, "val",   (args.img_h,args.img_w), norm)
    test_ds  = EyeClsDataset(args.data_root, "test",  (args.img_h,args.img_w), norm)
    tl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    te = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.arch, bool(args.pretrained)).to(device)

    # Compute pos_weight for BCE if imbalance exists
    n_pos = sum(1 for _,y in train_ds.items if y>0.5)
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    os.makedirs(args.out, exist_ok=True)
    best = {"AUROC": 0.0}; log = []
    for ep in range(1, args.epochs+1):
        model.train(); epoch_loss = 0.0
        for x, y, _ in tqdm(tl, desc=f"Epoch {ep}/{args.epochs}"):
            x = x.to(device, non_blocking=True); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x).squeeze(1)
                loss = bce(logits, y.squeeze(1))
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            epoch_loss += float(loss.detach().cpu())
        met,_,_ = eval_metrics(model, vl, device); sch.step()
        log.append({"epoch": ep, "train_loss": epoch_loss/len(tl), **met})
        print("Val:", met)
        torch.save({"model": model.state_dict(), "epoch": ep, "metrics": met}, os.path.join(args.out,"last.ckpt"))
        # Select best by AUROC (you can switch to AUPRC if desired)
        if met["AUROC"] > best["AUROC"]:
            best = met
            torch.save(model.state_dict(), os.path.join(args.out,"best.pt"))
            with open(os.path.join(args.out,"best_val_metrics.json"),"w") as f: f.write(json.dumps(best, indent=2))
        with open(os.path.join(args.out,"train_log.json"),"w") as f: f.write(json.dumps(log, indent=2))

    # Test
    # 1) 验证集分数
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device))
    vm, yv, sv = eval_metrics(model, vl, device)
    np.savez(os.path.join(args.out, "scores_val.npz"), y=yv, score=sv)

    # 2) 选阈值
    t_star = find_threshold(yv, sv, how='bac')

    # 3) 测试集 + 阈值化三指标
    tm, yt, st = eval_metrics(model, te, device)
    thr = eval_with_threshold(yt, st, t_star)

    # 4) 合并保存
    tm.update(thr)
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        f.write(json.dumps(tm, indent=2))
    np.savez(os.path.join(args.out, "scores_test.npz"), y=yt, score=st)
    print("Test:", tm)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_splits")
    ap.add_argument("--out", default="ckpts/cls")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18","efficientnet_b0"])
    ap.add_argument("--pretrained", type=int, default=0, help="1 to use ImageNet pretrained weights (needs internet for torchvision/timm weights)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, nargs=2, default=[128,256])
    args = ap.parse_args(); args.img_h,args.img_w = args.img_size
    main(args)
