# tools/vis_pred_masks_plus.py
# Visualize predicted masks with multiple modes: plain | heatmap | edge | grid
# Data layout: data_root/split/{open,closed}/*.{jpg,png,jpeg,bmp}
#
# Examples:
# 1) Plain overlay (per-image panels)
#   python tools/vis_pred_masks_plus.py --data_root data_splits --split test \
#     --ckpt ckpts/unet_cam/best.pt --img_size 128 256 --out vis_unet_cam --mode plain --th 0.5
#
# 2) Probability heatmap overlay
#   python tools/vis_pred_masks_plus.py --data_root data_splits --split test \
#     --ckpt ckpts/unet_cam/best.pt --img_size 128 256 --out vis_unet_cam --mode heatmap
#
# 3) Edge highlight (pred green, ref red)
#   python tools/vis_pred_masks_plus.py --data_root data_splits --split test \
#     --ckpt ckpts/unet_cam/best.pt --img_size 128 256 --out vis_unet_cam --mode edge \
#     --mask_root cam_masks --show_ref 1 --th 0.5
#
# 4) Batch grid (overlay only), first 48 samples into an 8x6 grid
#   python tools/vis_pred_masks_plus.py --data_root data_splits --split test \
#     --ckpt ckpts/unet_cam/best.pt --img_size 128 256 --out vis_unet_cam --mode grid \
#     --max_n 48 --grid_cols 8 --th 0.5
#
import argparse, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torchvision.transforms.functional import to_tensor

# allow importing project modules like seg.unet_tiny from project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from seg.unet_tiny import UNetTiny


# ----------------------------
# IO helpers
# ----------------------------
def load_model(ckpt_path, device):
    m = UNetTiny().to(device)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def list_images(root, split):
    items = []
    for cls in ["open", "closed"]:
        d = Path(root) / split / cls
        if not d.exists():
            continue
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            imgs += list(d.glob(ext))
        imgs = sorted(imgs)
        for p in imgs:
            items.append((p, cls))
    return items

def load_mask_if_exists(mask_root, split, cls, stem):
    if mask_root is None:
        return None
    p = Path(mask_root) / split / cls / (stem + ".png")
    if p.exists():
        return Image.open(p).convert("L")
    return None


# ----------------------------
# Visual helpers
# ----------------------------
def overlay_mask(img_rgb, mask_bin, color=(0, 255, 0), alpha=0.45):
    """
    img_rgb: PIL RGB
    mask_bin: numpy (H,W) in {0,1}
    """
    from PIL import Image as _Image
    overlay = _Image.new("RGBA", img_rgb.size, (0, 0, 0, 0))
    color_img = _Image.new("RGBA", img_rgb.size, (*color, int(255 * alpha)))
    am = _Image.fromarray((mask_bin * 255).astype(np.uint8), mode="L")
    overlay = _Image.composite(color_img, overlay, am)
    out = _Image.alpha_composite(img_rgb.convert("RGBA"), overlay).convert("RGB")
    return out

def prob_to_rgba(prob, cmap_name="viridis", alpha=0.45):
    """
    prob: (H,W) in [0,1]
    returns PIL RGBA heatmap image resized later if needed
    """
    H, W = prob.shape
    try:
        from matplotlib import cm
        cmap = cm.get_cmap(cmap_name)
        rgba = (cmap(prob) * 255).astype(np.uint8)  # (H,W,4)
        rgba[..., 3] = int(255 * alpha)
    except Exception:
        # fallback: simple green intensity
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 1] = (prob * 255).astype(np.uint8)  # G
        rgba[..., 3] = int(255 * alpha)
    return Image.fromarray(rgba, mode="RGBA")

def overlay_heatmap(img_rgb, prob, alpha=0.45, cmap="viridis"):
    hm = prob_to_rgba(prob, cmap_name=cmap, alpha=alpha)  # RGBA
    return Image.alpha_composite(img_rgb.convert("RGBA"), hm.resize(img_rgb.size, Image.BILINEAR)).convert("RGB")

def binary_edge(mask_bin):
    """
    4-neighborhood edge = mask XOR erode(mask)
    Erode via 4-neighbor intersection without external deps.
    """
    m = (mask_bin.astype(np.uint8) > 0).astype(np.uint8)
    # pad with zeros
    up    = np.pad(m[1:, :], ((0,1),(0,0)), mode="constant")
    down  = np.pad(m[:-1,:], ((1,0),(0,0)), mode="constant")
    left  = np.pad(m[:,1:],  ((0,0),(0,1)), mode="constant")
    right = np.pad(m[:,:-1], ((0,0),(1,0)), mode="constant")
    er = m & up & down & left & right
    edge = (m ^ er).astype(np.uint8)
    return edge

def overlay_edge(img_rgb, mask_bin, color=(0,255,0), thickness=1):
    """
    Draw thin edges of mask on top of image. thickness currently 1px.
    """
    edge = binary_edge(mask_bin)
    # build RGBA layer with only edge pixels colored
    rgba = np.zeros((img_rgb.height, img_rgb.width, 4), np.uint8)
    rgba[edge==1, 0] = color[0]
    rgba[edge==1, 1] = color[1]
    rgba[edge==1, 2] = color[2]
    rgba[edge==1, 3] = 255
    layer = Image.fromarray(rgba, mode="RGBA")
    return Image.alpha_composite(img_rgb.convert("RGBA"), layer).convert("RGB")

def panel(*imgs):
    """Horizontally concat PIL images."""
    w = sum(im.width for im in imgs)
    h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas

def make_grid(images, cols=6, pad=4, bg=(255,255,255)):
    """
    images: list of PIL RGB (same size)
    """
    if len(images)==0:
        return None
    W, H = images[0].size
    rows = int(np.ceil(len(images)/cols))
    grid = Image.new("RGB", (cols*W + (cols-1)*pad, rows*H + (rows-1)*pad), bg)
    for i,im in enumerate(images):
        r, c = divmod(i, cols)
        x = c*(W+pad); y = r*(H+pad)
        grid.paste(im, (x,y))
    return grid


# ----------------------------
# Main
# ----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.ckpt, device)
    items = list_images(args.data_root, args.split)
    if args.max_n > 0:
        items = items[: args.max_n]

    overlays_for_grid = []

    for p, cls in items:
        # read & resize
        im = Image.open(p).convert("RGB").resize((args.img_w, args.img_h), Image.BILINEAR)
        x = to_tensor(im).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
        m_bin = (prob >= args.th).astype(np.uint8)

        # optional ref mask
        ref_bin = None
        if args.mask_root is not None:
            stem = Path(p).stem
            gt = load_mask_if_exists(args.mask_root, args.split, cls, stem)
            if gt is not None:
                gt = gt.resize((args.img_w, args.img_h), Image.NEAREST)
                ref_bin = (np.array(gt) > 127).astype(np.uint8)

        if args.mode == "plain":
            vis_pred = overlay_mask(im, m_bin, color=(0,255,0), alpha=args.alpha)
            imgs = [im, vis_pred, Image.fromarray((m_bin*255).astype(np.uint8)).convert("RGB")]
            if args.show_ref and ref_bin is not None:
                vis_gt = overlay_mask(im, ref_bin, color=(255,0,0), alpha=args.alpha)
                imgs.insert(2, vis_gt)  # [orig, pred, gt, bin]
            out_img = panel(*imgs)
            # save per-image panel
            rel = Path(p).relative_to(Path(args.data_root)/args.split)
            out_dir = Path(args.out)/rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir/(rel.stem + "_vis_plain.jpg")
            out_img.save(out_path, quality=92)

        elif args.mode == "heatmap":
            vis_pred = overlay_heatmap(im, prob, alpha=args.alpha, cmap=args.cmap)
            imgs = [im, vis_pred]
            if args.show_ref and ref_bin is not None:
                vis_gt = overlay_mask(im, ref_bin, color=(255,0,0), alpha=args.alpha)
                imgs.append(vis_gt)
            out_img = panel(*imgs)
            rel = Path(p).relative_to(Path(args.data_root)/args.split)
            out_dir = Path(args.out)/rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir/(rel.stem + "_vis_heatmap.jpg")
            out_img.save(out_path, quality=92)

        elif args.mode == "edge":
            vis_pred = overlay_edge(im, m_bin, color=(0,255,0))
            if args.show_ref and ref_bin is not None:
                vis_pred = overlay_edge(vis_pred, ref_bin, color=(255,0,0))
            imgs = [im, vis_pred, Image.fromarray((m_bin*255).astype(np.uint8)).convert("RGB")]
            out_img = panel(*imgs)
            rel = Path(p).relative_to(Path(args.data_root)/args.split)
            out_dir = Path(args.out)/rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir/(rel.stem + "_vis_edge.jpg")
            out_img.save(out_path, quality=92)

        elif args.mode == "grid":
            # prepare a single overlay (use plain style by default)
            if args.grid_style == "plain":
                ov = overlay_mask(im, m_bin, color=(0,255,0), alpha=args.alpha)
            elif args.grid_style == "heatmap":
                ov = overlay_heatmap(im, prob, alpha=args.alpha, cmap=args.cmap)
            elif args.grid_style == "edge":
                ov = overlay_edge(im, m_bin, color=(0,255,0))
                if args.show_ref and ref_bin is not None:
                    ov = overlay_edge(ov, ref_bin, color=(255,0,0))
            else:
                ov = overlay_mask(im, m_bin, color=(0,255,0), alpha=args.alpha)
            overlays_for_grid.append(ov)

        else:
            raise ValueError("Unknown mode: " + str(args.mode))

    # If grid mode, save one big figure
    if args.mode == "grid":
        if len(overlays_for_grid) == 0:
            print("[WARN] No images to grid.")
        else:
            grid_img = make_grid(overlays_for_grid, cols=args.grid_cols, pad=args.grid_pad)
            out_path = Path(args.out)/f"grid_{args.grid_style}.jpg"
            grid_img.save(out_path, quality=92)
            print(f"[OK] Saved grid to {out_path}")

    print(f"[OK] Done. Outputs in: {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--ckpt", required=True, help="Path to UNet best.pt")
    ap.add_argument("--img_size", type=int, nargs=2, default=[128,256])
    ap.add_argument("--out", default="vis_masks_plus")
    ap.add_argument("--mask_root", default=None, help="Optional: reference masks root (e.g., cam_masks)")
    ap.add_argument("--max_n", type=int, default=0, help="0 means all")
    ap.add_argument("--th", type=float, default=0.5, help="threshold for binary mask in plain/edge/grid(plain)")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay transparency 0..1")
    ap.add_argument("--mode", default="plain", choices=["plain","heatmap","edge","grid"], help="visualization mode")
    # heatmap options
    ap.add_argument("--cmap", default="viridis", help="matplotlib cmap for heatmap mode")
    # grid options
    ap.add_argument("--grid_style", default="plain", choices=["plain","heatmap","edge"], help="overlay style for grid cells")
    ap.add_argument("--grid_cols", type=int, default=6)
    ap.add_argument("--grid_pad", type=int, default=4)
    # show reference mask (red) overlay when applicable
    ap.add_argument("--show_ref", type=int, default=0, help="1 to overlay reference mask (if provided)")
    args = ap.parse_args()
    args.img_h, args.img_w = args.img_size
    main(args)
