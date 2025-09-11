
# seg/train_unet_pseudo.py
# Train UNet with pseudo band masks (open: band; closed: zeros)
from torchvision.transforms.functional import to_tensor
import numpy as np, torch
from data_utils.pseudo_mask import band_mask

import argparse, os, json, numpy as np
from pathlib import Path
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seg.unet_tiny import UNetTiny
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix

def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2*(prob*target).sum(dim=(1,2,3))
    den = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return 1 - (num/den).mean()

def opening_ratio_from_mask(prob):
    mp = (prob>0.5).float()
    B, _, H, W = mp.shape
    heights = mp.squeeze(1).sum(1)  # (B,W)
    s = (torch.median(heights,1).values / float(H)).clamp(0,1)
    return s

def eval_image_level(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x, m, y, _ in loader:
            x = x.to(device); y = y.squeeze(1).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            s = opening_ratio_from_mask(prob)
            ys.append(y.detach().cpu().numpy()); ps.append(s.detach().cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {
        "AUROC": float(roc_auc_score(ys, ps)),
        "AUPRC": float(average_precision_score(ys, ps))
    }, ys, ps

def find_threshold(y, s, how='bac'):
    """
    在验证集上选阈值：
    how='bac'：最大化 Balanced Accuracy（推荐）
    how='f1' ：最大化 F1
    how='youden'：最大化 TPR+TNR-1（Youden's J）
    """
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

class SegDatasetPseudo(Dataset):
    def __init__(self, data_root, split, size_hw, band_ratio=0.12):
        self.paths = []
        for cls in ["open","closed"]:
            d = Path(data_root)/split/cls
            if not d.exists(): continue
            for p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.jpeg")) + list(d.glob("*.bmp"))):
                self.paths.append((p, 1.0 if cls=="open" else 0.0))
        self.H, self.W = size_hw
        self.band_ratio = band_ratio
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p, y = self.paths[idx]
        im = Image.open(p).convert("RGB").resize((self.W, self.H), Image.BILINEAR)
        x = to_tensor(im)  # 0..1
        if y > 0.5:
            m = band_mask(self.H, self.W, band_ratio=self.band_ratio).astype(np.float32)
        else:
            m = np.zeros((self.H,self.W), np.float32)
        m = torch.from_numpy(m)[None,:,:]
        y = torch.tensor([y], dtype=torch.float32)
        return x, m, y, str(p)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SegDatasetPseudo(args.data_root, "train", (args.img_h,args.img_w), band_ratio=args.band_ratio)
    val_ds   = SegDatasetPseudo(args.data_root, "val",   (args.img_h,args.img_w), band_ratio=args.band_ratio)
    test_ds  = SegDatasetPseudo(args.data_root, "test",  (args.img_h,args.img_w), band_ratio=args.band_ratio)
    tl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    te = DataLoader(test_ds,  batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetTiny().to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"AUPRC": 0.0}; os.makedirs(args.out, exist_ok=True); log = []
    for ep in range(1, args.epochs+1):
        model.train(); epoch_loss=0.0
        for x, m, y, _ in tqdm(tl, desc=f"Epoch {ep}/{args.epochs}"):
            x=x.to(device); m=m.to(device)
            logits=model(x)
            loss=0.5*bce(logits,m)+0.5*dice_loss(logits,m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            epoch_loss += float(loss.detach().cpu())
        met,_,_ = eval_image_level(model, vl, device); sch.step()
        log.append({"epoch": ep, "train_loss": epoch_loss/len(tl), **met})
        print("Val:", met)
        torch.save({"model":model.state_dict(),"epoch":ep,"metrics":met}, os.path.join(args.out,"last.ckpt"))
        if met["AUPRC"]>best["AUPRC"]:
            best=met; torch.save(model.state_dict(), os.path.join(args.out,"best.pt"))
            open(os.path.join(args.out,"best_val_metrics.json"),"w").write(json.dumps(best,indent=2))
        open(os.path.join(args.out,"train_log.json"),"w").write(json.dumps(log,indent=2))
    # test
    # 1) 用最优模型重新在验证集跑一遍，落盘 val 分数
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device))
    vm, yv, sv = eval_image_level(model, vl, device)
    np.savez(os.path.join(args.out, "scores_val.npz"), y=yv, score=sv)

    # 2) 在验证集选阈值（推荐用 Balanced Accuracy）
    t_star = find_threshold(yv, sv, how='bac')  # 可改 'f1' 看你论文偏好

    # 3) 在测试集评估（阈值固定为 t_star）
    tm, yt, st = eval_image_level(model, te, device)
    thr = eval_with_threshold(yt, st, t_star)

    # 4) 合并阈值化后的三项指标，并保存
    tm.update(thr)
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        f.write(json.dumps(tm, indent=2))

    # 5) 继续保存原始分数（便于 PR/ROC）
    np.savez(os.path.join(args.out, "scores_test.npz"), y=yt, score=st)
    print("Test:", tm)


if __name__=="__main__":
    import torch
    from torch.utils.data import DataLoader
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_splits")
    ap.add_argument("--out", default="ckpts/unet_pseudo")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, nargs=2, default=[128,256])
    ap.add_argument("--band_ratio", type=float, default=0.12)
    args=ap.parse_args(); args.img_h,args.img_w=args.img_size; main(args)
