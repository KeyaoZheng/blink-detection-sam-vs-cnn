import argparse, os, sys, time, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path: sys.path.append(ROOT_DIR)

from seg.unet_tiny import UNetTiny

def load_cls_model(arch, num_classes=1, pretrained=False):
    if arch == "resnet18":
        from torchvision import models
        import torch.nn as nn
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif arch == "efficientnet_b0":
        import timm
        m = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
        return m
    else:
        raise ValueError("arch must be resnet18 | efficientnet_b0")

def list_images(root, split):
    items=[]
    for cls in ["open","closed"]:
        d = Path(root)/split/cls
        if not d.exists(): continue
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            items += sorted(d.glob(ext))
    return items

@torch.no_grad()
def bench(model, device, batcher, iters=100, warmup=20, amp=False):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    times=[]
    scaler_ctx = torch.cuda.amp.autocast(enabled=amp) if device.type=="cuda" else torch.autocast("cpu", enabled=False)
    # warmup
    for _ in range(warmup):
        x = next(batcher)
        with scaler_ctx:
            _ = model(x.to(device, non_blocking=True))
        if device.type=="cuda": torch.cuda.synchronize()
    # timed
    for _ in range(iters):
        x = next(batcher)
        t0 = time.perf_counter()
        with scaler_ctx:
            _ = model(x.to(device, non_blocking=True))
        if device.type=="cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1-t0)
    times = np.array(times)
    mem_mb = float(torch.cuda.max_memory_allocated()/1024/1024) if device.type=="cuda" else 0.0
    return {
        "latency_mean_ms": float(times.mean()*1000),
        "latency_median_ms": float(np.median(times)*1000),
        "latency_p95_ms": float(np.percentile(times,95)*1000),
        "throughput_images_per_s": float((batcher.bs)/times.mean()),
        "gpu_mem_peak_mb": mem_mb,
        "iters": iters, "warmup": warmup, "batch_size": batcher.bs
    }

class RandomBatcher:
    def __init__(self, bs, h, w, ch=3, device="cpu"):
        self.bs, self.h, self.w, self.ch = bs, h, w, ch,
    def __iter__(self): return self
    def __next__(self):
        return torch.rand(self.bs, self.ch, self.h, self.w)

class ImageBatcher:
    def __init__(self, paths, bs, h, w):
        self.paths = paths; self.bs=bs; self.h=h; self.w=w; self.i=0
    def __iter__(self): return self
    def __next__(self):
        if self.i >= len(self.paths):
            self.i = 0
        batch = []
        for _ in range(self.bs):
            p = self.paths[self.i]
            im = Image.open(p).convert("RGB").resize((self.w,self.h), Image.BILINEAR)
            batch.append(to_tensor(im))
            self.i = (self.i+1) % len(self.paths)
        return torch.stack(batch,0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["seg","cls"], required=True)
    ap.add_argument("--arch", default="unet", help="seg: unet | cls: resnet18/efficientnet_b0")
    ap.add_argument("--ckpt", default=None, help="optional: load weights (best.pt)")
    ap.add_argument("--data_root", default=None, help="if set, use real images from data_root/split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--img_size", type=int, nargs=2, default=[128,256])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--amp", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h,w = args.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # build model
    if args.task=="seg":
        model = UNetTiny()
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
            model.load_state_dict(sd, strict=False)
    else:
        model = load_cls_model("resnet18" if args.arch=="unet" else args.arch, num_classes=1, pretrained=False)
        if args.ckpt:
            sd = torch.load(args.ckpt, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
            model.load_state_dict(sd, strict=False)
    model.eval().to(device)

    # batcher
    if args.data_root and Path(args.data_root).exists():
        paths = list_images(args.data_root, args.split)
        assert len(paths)>0, "No images found under data_root/split"
        batcher = ImageBatcher(paths, args.bs, h, w)
    else:
        batcher = RandomBatcher(args.bs, h, w)

    stats = bench(model, device, iter(batcher), iters=args.iters, warmup=args.warmup, amp=bool(args.amp))
    stats.update({
        "task": args.task, "arch": args.arch, "device": str(device),
        "img_h": h, "img_w": w
    })
    print(json.dumps(stats, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        open(args.out, "w").write(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
