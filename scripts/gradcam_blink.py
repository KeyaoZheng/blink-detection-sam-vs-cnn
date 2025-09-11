# gradcam_blink.py
import argparse, os, inspect
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from PIL import Image
from collections import OrderedDict

import timm
import torchvision.models as tvm

# pytorch-grad-cam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------- Utils: safe torch.load on PyTorch 2.6 ----------
def safe_load_checkpoint(path, map_location="cpu"):
    """Load checkpoint under PyTorch 2.6 (weights_only default True).
       Falls back to weights_only=False when needed."""
    import numpy as np
    try:
        # allow numpy scalar in safe mode if needed
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # if you trust the checkpoint file, this will work
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    return ckpt

def extract_state_dict(ckpt):
    """Get state_dict from various checkpoint formats."""
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt if isinstance(ckpt, dict) else ckpt

def strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
    return new_sd

def detect_impl_from_keys(keys):
    """Very simple heuristic: torchvision vs timm by param names."""
    keys = list(keys)
    if any(k.startswith("features.") or k.startswith("classifier.") for k in keys):
        return "torchvision"
    if any(k.startswith("conv_stem") or k.startswith("blocks.") or k.startswith("bn1.") for k in keys):
        return "timm"
    # default to timm for modern backbones
    return "timm"

# ---------- Build model matching the checkpoint impl ----------
def build_model_matched(arch: str, impl_hint: str, num_classes: int = 2,
                        pretrained: bool = False, state_keys=None):
    a = arch.lower()

    # Some architectures we prefer fixed sources for stability
    if a == "resnet18":
        # use torchvision for resnet18
        m = tvm.resnet18(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    # EfficientNet-B0: timm variant matches your checkpoint with conv_stem/blocks.*
    if a == "efficientnet_b0":
        if impl_hint == "timm":
            return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
        else:
            m = tvm.efficientnet_b0(weights="DEFAULT" if pretrained else None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
            return m

    # MobileNetV2
    if a in ("mobilenetv2_100", "mobilenet_v2"):
        if impl_hint == "torchvision":
            m = tvm.mobilenet_v2(weights="DEFAULT" if pretrained else None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
            return m
        else:
            return timm.create_model("mobilenetv2_100", pretrained=pretrained, num_classes=num_classes)

    # ConvNeXt-Tiny (timm)
    if a == "convnext_tiny":
        def _prefer_timm():
            if state_keys:
                return any(k.startswith(("stages.", "stem.")) for k in state_keys)
            return (impl_hint == "timm")

        def _prefer_tv():
            if state_keys:
                return any(k.startswith(("features.", "0.", "1.", "2.", "3.")) for k in state_keys)
            return (impl_hint == "torchvision")

        if _prefer_timm():
            return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)
        elif _prefer_tv():
            m = tvm.convnext_tiny(weights="DEFAULT" if pretrained else None)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
            return m
        else:
            return timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes)

    # ShuffleNetV2 (torchvision)
    if a in ("shufflenet_v2_x1_0", "shufflenetv2_x1_0"):
        m = tvm.shufflenet_v2_x1_0(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    # ViT & Swin (timm)
    if a in ("vit_tiny_patch16_224", "swin_tiny_patch4_window7_224"):
        return timm.create_model(a, pretrained=pretrained, num_classes=num_classes)

    # fallback: try timm
    try:
        return timm.create_model(a, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        pass
    # fallback: try torchvision direct ctor if exists
    if hasattr(tvm, a):
        ctor = getattr(tvm, a)
        m = ctor(weights="DEFAULT" if pretrained else None)
        # attempt to swap head
        if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif hasattr(m, "classifier"):
            # try last linear
            if isinstance(m.classifier, nn.Sequential) and isinstance(m.classifier[-1], nn.Linear):
                m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    raise RuntimeError(f"Unknown model arch: {arch}")

def load_weights_to_model(model, state_dict):
    sd = strip_module_prefix(state_dict) if isinstance(state_dict, dict) else state_dict
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:   print("  missing (head等通常无碍):", missing[:10], "...")
        if unexpected:print("  unexpected:", unexpected[:10], "...")
    return model

# ---------- Pick target layers ----------
def pick_target_layers(model, arch: str):
    n = arch.lower()
    m = model
    # ResNet-18 (torchvision)
    if "resnet18" in n:
        return [m.layer4[-1].conv2]
    # MobileNetV2 (tv: features; timm: find last Conv2d)
    if "mobilenet" in n:
        if hasattr(m, "features"):
            return [m.features[-1]]
    # EfficientNet-B0
    if "efficientnet_b0" in n:
        if hasattr(m, "conv_head"):   # timm
            return [m.conv_head]
        if hasattr(m, "features"):    # torchvision
            return [m.features[-1]]
    # ShuffleNetV2 (torchvision)
    if "shufflenet" in n and hasattr(m, "conv5"):
        return [m.conv5]
    # Swin / ViT: give a layer; EigenCAM recommended
    if "swin_tiny" in n and hasattr(m, "layers"):
        return [m.layers[-1].blocks[-1].norm2]
    if "vit_tiny_patch16_224" in n and hasattr(m, "blocks"):
        return [m.blocks[-1].norm1]
    # ConvNeXt-Tiny (timm)
    if "convnext_tiny" in n:
        last_stage = getattr(m, "stages", None) or getattr(m, "features", None)
        if last_stage is not None:
            stage_last = last_stage[-1]
            last_block = stage_last.blocks[-1] if hasattr(stage_last, "blocks") else stage_last
            for attr in ("dwconv", "conv_dw", "depthwise_conv", "conv"):
                if hasattr(last_block, attr):
                    layer = getattr(last_block, attr)
                    if isinstance(layer, torch.nn.Conv2d):
                        return [layer]
        for layer in reversed(list(m.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                return [layer]

        raise RuntimeError("No suitable target layer found.")


    for layer in reversed(list(m.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return [layer]
    raise RuntimeError("No suitable target layer found.")
    # Fallback: last Conv2d
    for layer in reversed(list(m.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return [layer]
    raise RuntimeError("No suitable target layer found; specify manually via code if needed.")



# ---------- Preprocess ----------
def build_transform(image_size=224):
    mean=(0.485,0.456,0.406); std=(0.229,0.224,0.225)
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def load_rgb_image(path):
    return np.array(Image.open(path).convert("RGB"))

# ---------- Grad-CAM runner (version compatible) ----------
def make_cam_instance(cam_cls, model, target_layers, device):
    # pytorch-grad-cam API: old has use_cuda, new has device
    sig = inspect.signature(cam_cls)
    kwargs = {"model": model, "target_layers": target_layers}
    if "device" in sig.parameters:
        kwargs["device"] = device
    elif "use_cuda" in sig.parameters:
        kwargs["use_cuda"] = (device == "cuda")
    return cam_cls(**kwargs)

def run_cam_on_image(model, target_layers, rgb_np, transform, target_class=None,
                     cam_type="gradcam", device="cpu", save_triple=False):
    model = model.to(device).eval()
    H, W = rgb_np.shape[:2]
    rgb_norm = rgb_np.astype(np.float32) / 255.0
    tensor = transform(Image.fromarray(rgb_np)).unsqueeze(0).to(device)

    cam_map = None
    cam_cls_map = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "eigen": EigenCAM}
    cam_cls = cam_cls_map[cam_type]

    with make_cam_instance(cam_cls, model, target_layers, device) as cam:
        targets = [ClassifierOutputTarget(int(target_class))] if target_class is not None else None
        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        cam_map = grayscale_cam[0] if grayscale_cam.ndim == 3 else grayscale_cam

    # resize CAM to input image size
    cam_map = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)
    # 强制 0-1 归一化，避免常数图导致纯蓝
    cam_min, cam_max = float(cam_map.min()), float(cam_map.max())
    if cam_max - cam_min < 1e-12:
        cam_map = np.zeros_like(cam_map, dtype=np.float32)
    else:
        cam_map = (cam_map - cam_min) / (cam_max - cam_min)

    vis = show_cam_on_image(rgb_norm, cam_map, use_rgb=True)

    if save_triple:
        # build a panel: [original | heatmap | overlay]
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        panel = np.concatenate([rgb_np, heatmap, vis], axis=1)
        return vis, cam_map, panel
    return vis, cam_map, None

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18",
                    help="resnet18 | mobilenetv2_100 | mobilenet_v2 | efficientnet_b0 | convnext_tiny | shufflenet_v2_x1_0 | vit_tiny_patch16_224 | swin_tiny_patch4_window7_224")
    ap.add_argument("--weights", type=str, required=True, help="path to checkpoint .pth")
    ap.add_argument("--image", type=str, default=None, help="an image path")
    ap.add_argument("--folder", type=str, default=None, help="a folder of images (recursive)")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--cam", type=str, default="gradcam", choices=["gradcam","gradcam++","eigen"])
    ap.add_argument("--force_class", type=int, default=None, help="set a fixed class id for CAM (e.g., 1 = blink). If None, use predicted class.")
    ap.add_argument("--out_dir", type=str, default="cam_out")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--save_triple", action="store_true", help="also save triptych: [original|heatmap|overlay]")
    args = ap.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load checkpoint & detect implementation
    ckpt = safe_load_checkpoint(args.weights, map_location=device)
    state = extract_state_dict(ckpt)
    impl = detect_impl_from_keys(state.keys())
    state_keys = list(state.keys())
    print(f"[INFO] Detected checkpoint impl: {impl}")

    # 2) build model & load weights
    print(f"[INFO] Building model: {args.model}")
    model = build_model_matched(args.model, impl_hint=impl, num_classes=2,
                                pretrained=False, state_keys=state_keys)  # ← 新增参数
    print(f"[INFO] Loading weights: {args.weights}")
    model = load_weights_to_model(model, state)

    # 3) target layers
    target_layers = pick_target_layers(model, args.model)
    print(f"[INFO] Target layers: {[type(l).__name__ for l in target_layers]}")

    # 4) image list
    paths = []
    if args.image: paths.append(args.image)
    if args.folder:
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            paths += list(Path(args.folder).rglob(ext))
        paths = [str(p) for p in paths]
    assert paths, "Provide --image or --folder"

    transform = build_transform(args.image_size)

    # 5) run
    for p in paths:
        rgb = load_rgb_image(p)
        vis, cam_map, panel = run_cam_on_image(
            model, target_layers, rgb, transform,
            target_class=args.force_class, cam_type=args.cam,
            device=device, save_triple=args.save_triple
        )
        base = f"{Path(p).stem}_{args.model}_{args.cam}"
        out_path = out_dir / f"{base}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")
        if panel is not None:
            out_tri = out_dir / f"{base}_tri.png"
            cv2.imwrite(str(out_tri), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            print(f"Saved: {out_tri}")

if __name__ == "__main__":
    main()


