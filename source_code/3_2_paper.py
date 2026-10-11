"""
Reproduce Figure 3 (class-specific visualizations) of the survey that cites
Chefer et al. 2021a, "Transformer Interpretability Beyond Attention Visualization" (CVPR 2021).

Uses the OFFICIAL implementation (hila-chefer/Transformer-Explainability, cloned to ./official_repo)
so every column is computed exactly as the authors do:

  rollout          -> Baselines.generate_rollout          (ViT_new,      Abnar & Zuidema)
  raw-attention    -> LRP.generate_LRP("last_layer_attn") (ViT_LRP,      last-layer CLS attention)
  GradCAM          -> Baselines.generate_cam_attn         (ViT_new,      Grad-CAM on last attention map)
  LRP              -> LRP.generate_LRP("full")            (ViT_orig_LRP, full LRP to the pixels)
  partial LRP      -> LRP.generate_LRP("last_layer")      (ViT_orig_LRP, LRP of last attention layer)
  Beyond Attention -> LRP.generate_LRP("transformer_attribution")  (ViT_LRP, proposed method)

Only change w.r.t. the official code: .cuda() calls are replaced so it runs on CPU.
"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "official_repo")
sys.path.insert(0, REPO)
os.chdir(REPO)  # the repo uses imports relative to its root

from baselines.ViT.ViT_explanation_generator import Baselines  # noqa: E402
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP  # noqa: E402
from baselines.ViT.ViT_new import vit_base_patch16_224 as vit_new  # noqa: E402
from baselines.ViT.ViT_orig_LRP import vit_base_patch16_224 as vit_orig_LRP  # noqa: E402

DEVICE = torch.device("cpu")
IMG_DIR = os.path.join(HERE, "images")
OUT_DIR = os.path.join(HERE, "outputs")

# (image file, [(row label, ImageNet class index or None = model's top-1)])
# ImageNet ids: 243 bull mastiff, 282 tiger cat, 101 tusker, 340 zebra  (same ids as the authors' notebook)
EXAMPLES = [
    ("dog_and_cat.png", [("Dog", 243), ("Cat", 282)]),
    ("elephant_and_zebra.png", [("Elephant", 101), ("Zebra", 340)]),
]
COLUMNS = ["rollout", "raw-attention", "GradCAM", "LRP", "partial LRP", "Beyond Attention"]

# ViT-B/16 (Google/timm weights) expects mean=std=0.5; whole image is resized to 224x224 (as in the figure)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def generate_LRP(model, x, index, method, start_layer=0, is_ablation=False):
    """CPU version of LRP.generate_LRP from the official repo."""
    model.eval()
    output = model(x)
    if index is None:
        index = int(output.argmax(-1))
    one_hot = np.zeros((1, output.size(-1)), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = torch.tensor(one_hot)
    score = (torch.from_numpy(one_hot).requires_grad_(True) * output).sum()
    model.zero_grad()
    score.backward(retain_graph=True)
    return model.relprop(one_hot_vector, method=method, is_ablation=is_ablation,
                         start_layer=start_layer, alpha=1)


def generate_cam_attn(model, x, index):
    """CPU version of Baselines.generate_cam_attn (Grad-CAM on last-layer attention)."""
    model.eval()
    output = model(x, register_hook=True)
    one_hot = np.zeros((1, output.size(-1)), dtype=np.float32)
    one_hot[0, index] = 1
    score = (torch.from_numpy(one_hot).requires_grad_(True) * output).sum()
    model.zero_grad()
    score.backward(retain_graph=True)
    grad = model.blocks[-1].attn.get_attn_gradients()
    cam = model.blocks[-1].attn.get_attention_map()
    cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
    grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
    grad = grad.mean(dim=[1, 2], keepdim=True)
    cam = (cam * grad).mean(0).clamp(min=0)
    return (cam - cam.min()) / (cam.max() - cam.min())


def to_heatmap(res, upsample=True):
    """Official post-processing: bilinear x16 upsample, min-max normalise."""
    res = res.detach()
    if upsample:
        res = F.interpolate(res.reshape(1, 1, 14, 14), scale_factor=16, mode="bilinear")
    res = res.reshape(224, 224).cpu().numpy()
    return (res - res.min()) / (res.max() - res.min())


def overlay(img01, mask):
    """Official show_cam_on_image: JET heatmap + image, rescaled by max."""
    heat = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heat = np.float32(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)) / 255
    cam = heat + np.float32(img01)
    return cam / np.max(cam)


def top5(model, x, names=None):
    p = torch.softmax(model(x), dim=1)[0].detach()  # the repo's forward registers grad hooks, so no no_grad()
    vals, idx = p.topk(5)
    return [(int(i), float(v)) for v, i in zip(vals, idx)]


def main():
    print("Loading models (ViT-B/16 pretrained weights are downloaded on first run)...")
    m_new = vit_new(pretrained=True).to(DEVICE).eval()
    m_lrp = vit_LRP(pretrained=True).to(DEVICE).eval()
    m_orig = vit_orig_LRP(pretrained=True).to(DEVICE).eval()
    baselines = Baselines(m_new)

    rows = []  # (input_img, row_label, [heatmap overlays])
    for fname, classes in EXAMPLES:
        pil = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
        x = transform(pil).unsqueeze(0).to(DEVICE)
        img01 = np.asarray(pil.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255
        print(f"\n{fname}: top-5 (class id, prob) = {top5(m_lrp, x)}")

        for label, cls in classes:
            print(f"  -> {label} (class {cls})")
            maps = [
                to_heatmap(baselines.generate_rollout(x, start_layer=0)),
                to_heatmap(generate_LRP(m_lrp, x, cls, "last_layer_attn")),
                to_heatmap(generate_cam_attn(m_new, x, cls)),
                to_heatmap(generate_LRP(m_orig, x, cls, "full"), upsample=False),
                to_heatmap(generate_LRP(m_orig, x, cls, "last_layer")),
                to_heatmap(generate_LRP(m_lrp, x, cls, "transformer_attribution", start_layer=0)),
            ]
            rows.append((img01, label, [overlay(img01, m) for m in maps]))

    # ---- also save every single heat map (used as thumbnails on the slides)
    panel_dir = os.path.join(OUT_DIR, "panels")
    os.makedirs(panel_dir, exist_ok=True)
    for img01, label, overlays in rows:
        for name, ov in zip(COLUMNS, overlays):
            plt.imsave(os.path.join(panel_dir, "%s_%s.png" % (label, name.replace(" ", "_"))), ov)

    # ---- figure in the layout of the paper: input | 6 methods, 2 rows per image
    fig, axes = plt.subplots(len(rows), 7, figsize=(14, 2.2 * len(rows)), dpi=200)
    for r, (img01, label, overlays) in enumerate(rows):
        axes[r, 0].imshow(img01)
        axes[r, 0].set_ylabel(label + " ->", rotation=0, ha="right", va="center", fontsize=11)
        for c, ov in enumerate(overlays, start=1):
            axes[r, c].imshow(ov)
        for c in range(7):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)
    axes[0, 0].set_title("Input", fontsize=11)
    for c, name in enumerate(COLUMNS, start=1):
        axes[0, c].set_title(name, fontsize=11)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "output_paper_figure3.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
