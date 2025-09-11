import argparse, os, time, random, json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support

# -----------------------
# Model zoo
# -----------------------
MODEL_CHOICES = [
    "mobilenetv2_100",
    "shufflenetv2_x1_0",
    "efficientnet_b0",
    "resnet18",
    "convnext_tiny",
    "vit_tiny_patch16_224",
    "swin_tiny_patch4_window7_224",
]

import timm
import torchvision.models as tvm

def build_model(name: str, num_classes: int = 2, pretrained: bool = True):
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        pass

    tv_map = {
        "shufflenetv2_x1_0": "shufflenet_v2_x1_0",
        "resnet18": "resnet18",
        "mobilenetv2_100": "mobilenet_v2",   # torchvision 里是 mobilenet_v2
        "efficientnet_b0": "efficientnet_b0",
    }
    if name in tv_map:
        ctor = getattr(tvm, tv_map[name])
        model = ctor(weights="DEFAULT" if pretrained else None)
        # 改分类头到 2 类
        if name.startswith("resnet"):
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
        elif "mobilenet" in name:
            in_f = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_f, num_classes)
        elif "shufflenet" in name:
            in_f = model.fc.in_features
            model.fc = nn.Linear(in_f, num_classes)
        elif "efficientnet_b0" in name:
            in_f = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_f, num_classes)
        return model

    raise RuntimeError(f"Unknown model name: {name}. "
                       f"Try one of timm models or torchvision equivalents.")

# -----------------------
# Repro
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------
# Metrics
# -----------------------
def metric_report(y_true, y_prob):
    # y_prob = P(y=1)  (positive = blink/Closed_Eyes)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float('nan')
    return {"acc": acc, "precision": p, "recall": r, "f1": f1, "auc": auc}

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# -----------------------
# Datasets & loaders
# -----------------------
class LabelMapDataset(Dataset):
    """
    Wrap an ImageFolder dataset and remap labels:
      Closed_Eyes -> 1 (blink, positive)
      Open_Eyes   -> 0 (non-blink, negative)
    """
    def __init__(self, base: datasets.ImageFolder, pos_name: str = "Closed_Eyes"):
        self.base = base
        self.pos_name = pos_name
        # map class idx to 0/1
        self.class_to_idx = base.class_to_idx  # e.g. {'Closed_Eyes': 0, 'Open_Eyes': 1} (alphabetical by default)
        if pos_name not in self.class_to_idx:
            # back-up: try 'blink'
            if "blink" in self.class_to_idx:
                self.pos_name = "blink"
            else:
                raise ValueError(f"Positive class '{pos_name}' not found in classes {list(self.class_to_idx.keys())}")
        self.pos_idx = self.class_to_idx[self.pos_name]

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, y_orig = self.base[idx]
        y = 1 if y_orig == self.pos_idx else 0
        return img, torch.tensor(y, dtype=torch.long)

def stratified_indices_by_label(labels: List[int], val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    n0_val = int(len(idx0) * val_ratio)
    n1_val = int(len(idx1) * val_ratio)
    val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]])
    train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]])
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx.tolist(), val_idx.tolist()

def build_loaders_from_imagefolder(root: Path, image_size: int, batch_size: int, num_workers: int, seed: int):
    # transforms
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # train
    train_dir = root / "train"
    val_dir   = root / "val"
    test_dir  = root / "test"

    has_val = val_dir.exists()

    base_train = datasets.ImageFolder(train_dir, transform=train_tf)
    if "Closed_Eyes" not in base_train.class_to_idx and "blink" not in base_train.class_to_idx:
        raise ValueError(f"Expected classes to include 'Closed_Eyes' (or 'blink'), got {base_train.class_to_idx}")

    train_wrapped = LabelMapDataset(base_train, pos_name="Closed_Eyes")

    if has_val:
        base_val = datasets.ImageFolder(val_dir, transform=eval_tf)
        val_wrapped = LabelMapDataset(base_val, pos_name="Closed_Eyes")
        train_ds = train_wrapped
        val_ds = val_wrapped
    else:
        labels = []
        for _, y in base_train.samples:
            y_bin = 1 if y == train_wrapped.pos_idx else 0
            labels.append(y_bin)
        tr_idx, va_idx = stratified_indices_by_label(labels, val_ratio=0.1, seed=seed)
        train_ds = Subset(train_wrapped, tr_idx)
        val_ds   = Subset(train_wrapped, va_idx)

    if not test_dir.exists():
        raise ValueError(f"Expected a 'test' folder under {root}")
    base_test = datasets.ImageFolder(test_dir, transform=eval_tf)
    test_wrapped = LabelMapDataset(base_test, pos_name="Closed_Eyes")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# -----------------------
# Train / Eval
# -----------------------
from torch.cuda.amp import GradScaler, autocast

def run_epoch(model, loader, criterion, optimizer=None, device="cuda", amp=True):
    is_train = optimizer is not None
    model.train(is_train)
    scaler = GradScaler(enabled=(amp and is_train))

    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(amp and is_train)):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = metric_report(all_labels, all_probs)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="ImageFolder root with train/[val]/test")
    parser.add_argument("--model", type=str, default="mobilenetv2_100", choices=MODEL_CHOICES)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # Build model
    model = build_model(args.model, num_classes=2, pretrained=not args.no_pretrained).to(device)

    # Build loaders (自动分出 val)
    train_loader, val_loader, test_loader = build_loaders_from_imagefolder(
        Path(args.data_dir), args.image_size, args.batch_size, args.num_workers, args.seed
    )

    # Loss / Optim / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state = -1.0, None
    history = {"train": [], "val": [], "test": None}
    patience, patience_counter = 10, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device=device, amp=args.amp)
        val_loss, val_m     = run_epoch(model, val_loader,   criterion, None,      device=device, amp=False)
        scheduler.step()

        history["train"].append({"epoch": epoch, "loss": train_loss, **train_m})
        history["val"].append({"epoch": epoch, "loss": val_loss, **val_m})

        print(f"[{epoch:03d}] Train L={train_loss:.4f} F1={train_m['f1']:.4f} Acc={train_m['acc']:.4f} "
              f"| Val L={val_loss:.4f} F1={val_m['f1']:.4f} Acc={val_m['acc']:.4f} AUC={val_m['auc']:.4f} "
              f"| {time.time()-t0:.1f}s")

        # early stop on val F1
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_m,
                "args": vars(args),
            }
            torch.save(best_state, os.path.join(args.out_dir, f"best_{args.model}.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Test with best
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    test_loss, test_m = run_epoch(model, test_loader, criterion, None, device=device, amp=False)
    history["test"] = {"loss": test_loss, **test_m}
    print(f"[TEST] L={test_loss:.4f} F1={test_m['f1']:.4f} Acc={test_m['acc']:.4f} AUC={test_m['auc']:.4f}")

    # Save history
    save_json(history, os.path.join(args.out_dir, f"history_{args.model}.json"))

if __name__ == "__main__":
    main()

