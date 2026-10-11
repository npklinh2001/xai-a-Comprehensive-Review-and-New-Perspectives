import os
from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T


class VisualExplanationResult(NamedTuple):
    method_name: str
    attribution_map: np.ndarray
    foreground_score: float
    background_score: float
    contrast_ratio: float
    class_specific: bool


class CustomAttentionModule(nn.Module):
    def __init__(self, original_attention: nn.Module, embed_dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.in_proj_weight = original_attention.in_proj_weight
        self.in_proj_bias = original_attention.in_proj_bias
        self.out_proj = original_attention.out_proj

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        # // Linear projection of Query, Key, Value vectors
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # // Scaled dot-product self-attention computation: Softmax(Q * K^T / sqrt(d_k))
        scaling = 1.0 / (self.head_dim ** 0.5)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scaling
        attention_probs = F.softmax(attention_scores, dim=-1)

        if attention_probs.requires_grad:
            attention_probs.retain_grad()

        attended_values = torch.matmul(attention_probs, values).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attended_values)
        return output, attention_probs


class ViTModelWrapper:
    def __init__(self) -> None:
        self.weights = models.ViT_B_16_Weights.DEFAULT
        self.model = models.vit_b_16(weights=self.weights).eval()
        self.categories = self.weights.meta["categories"]
        self.num_layers = len(self.model.encoder.layers)
        self.patch_blocks()

    def patch_blocks(self) -> None:
        for block in self.model.encoder.layers:
            block.custom_attention = CustomAttentionModule(block.self_attention)
            def create_forward(b):
                def forward_hook(x_in: torch.Tensor) -> torch.Tensor:
                    normed = b.ln_1(x_in)
                    out, attn_map = b.custom_attention(normed)
                    b.saved_attention = attn_map
                    b.saved_tokens = x_in
                    x_res = b.dropout(out) + x_in
                    y = b.mlp(b.ln_2(x_res))
                    return x_res + y
                return forward_hook
            block.forward = create_forward(block)


class IExplanationAlgorithm(ABC):
    @abstractmethod
    def compute_attribution(
        self,
        model_wrapper: ViTModelWrapper,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        pass


class RawAttentionAlgorithm(IExplanationAlgorithm):
    def compute_attribution(
        self,
        model_wrapper: ViTModelWrapper,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        with torch.no_grad():
            _ = model_wrapper.model(input_tensor)
        # // Extract CLS token row from final layer attention averaged across heads
        last_block = model_wrapper.model.encoder.layers[-1]
        attn = last_block.saved_attention[0].mean(dim=0)
        cls_to_patches = attn[0, 1:]
        normalized_scores = cls_to_patches / (cls_to_patches.max() + 1e-8)
        return normalized_scores.detach()


class AttentionRolloutAlgorithm(IExplanationAlgorithm):
    def compute_attribution(
        self,
        model_wrapper: ViTModelWrapper,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        with torch.no_grad():
            _ = model_wrapper.model(input_tensor)
        # // Attention Rollout: recursive multiplication of augmented attention matrices A_hat = 0.5 * I + 0.5 * A
        seq_len = 197
        accumulated_rollout = torch.eye(seq_len)
        for block in model_wrapper.model.encoder.layers:
            mean_heads = block.saved_attention[0].mean(dim=0)
            augmented = 0.5 * torch.eye(seq_len) + 0.5 * mean_heads
            augmented = augmented / augmented.sum(dim=-1, keepdim=True)
            accumulated_rollout = torch.matmul(augmented, accumulated_rollout)
        cls_rollout = accumulated_rollout[0, 1:]
        normalized_scores = cls_rollout / (cls_rollout.max() + 1e-8)
        return normalized_scores.detach()


class ViTGradCAMAlgorithm(IExplanationAlgorithm):
    def compute_attribution(
        self,
        model_wrapper: ViTModelWrapper,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        model_wrapper.model.zero_grad()
        target_layer = model_wrapper.model.encoder.layers[-2]
        saved_reps: List[torch.Tensor] = []

        def forward_capture_hook(m, inp, outp):
            inp[0].retain_grad()
            saved_reps.append(inp[0])

        hook_handle = target_layer.register_forward_hook(forward_capture_hook)
        inp = input_tensor.clone().requires_grad_(True)
        logits = model_wrapper.model(inp)
        score = logits[0, target_class]
        score.backward()
        hook_handle.remove()

        tokens = saved_reps[0]
        grad_tokens = tokens.grad

        # // Grad-CAM channel weights: mean gradient across spatial patch tokens
        patch_tokens = tokens[:, 1:]
        patch_grads = grad_tokens[:, 1:]
        channel_weights = patch_grads.mean(dim=1, keepdim=True)
        # // Linear combination followed by ReLU rectification
        cam = F.relu((channel_weights * patch_tokens).sum(dim=-1)).squeeze(0)
        normalized_scores = cam / (cam.max() + 1e-8)
        return normalized_scores.detach()


class FullLRPAlgorithm(IExplanationAlgorithm):
    def compute_attribution(
        self,
        model_wrapper: ViTModelWrapper,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        model_wrapper.model.zero_grad()
        inp = input_tensor.clone().requires_grad_(True)
        logits = model_wrapper.model(inp)
        score = logits[0, target_class]
        score.backward()

        # // Deep Taylor LRP approximation at input layer: R_i = ReLU(x_i * (d_Score / d_x_i))
        pixel_relevance = (inp * inp.grad).sum(dim=1, keepdim=True)
        pixel_relevance = F.relu(pixel_relevance)
        # // Average pool over 16x16 patch kernels to obtain patch relevance grid
        patch_relevance = F.avg_pool2d(pixel_relevance, kernel_size=16, stride=16).flatten()
        normalized_scores = patch_relevance / (patch_relevance.max() + 1e-8)
        return normalized_scores.detach()


class PartialLRPAlgorithm(IExplanationAlgorithm):
    def compute_attribution(
        self,
        model_wrapper: ViTModelWrapper,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        model_wrapper.model.zero_grad()
        inp = input_tensor.clone().requires_grad_(True)
        logits = model_wrapper.model(inp)
        score = logits[0, target_class]
        score.backward()

        # // Partial LRP: isolate relevance at final self-attention tensor R(A) = |A * d_Score / d_A|
        last_block = model_wrapper.model.encoder.layers[-1]
        attn = last_block.saved_attention
        grad = attn.grad
        relevance_attn = (attn * grad).abs().mean(dim=1).squeeze(0)
        cls_scores = relevance_attn[0, 1:]
        normalized_scores = cls_scores / (cls_scores.max() + 1e-8)
        return normalized_scores.detach()


class BeyondAttentionAlgorithm(IExplanationAlgorithm):
    def compute_attribution(
        self,
        model_wrapper: ViTModelWrapper,
        input_tensor: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        model_wrapper.model.zero_grad()
        inp = input_tensor.clone().requires_grad_(True)
        logits = model_wrapper.model(inp)
        score = logits[0, target_class]
        score.backward()

        # // Beyond Attention (Chefer et al., 2021): recursive propagation of A_bar = I + E_h((Grad_A * A)^+)
        seq_len = 197
        relevance_matrix = torch.eye(seq_len)
        for block in model_wrapper.model.encoder.layers:
            attn = block.saved_attention
            grad = attn.grad
            positive_relevance = F.relu(grad * attn).mean(dim=1).squeeze(0)
            augmented = torch.eye(seq_len) + positive_relevance
            augmented = augmented / (augmented.sum(dim=-1, keepdim=True) + 1e-8)
            relevance_matrix = torch.matmul(augmented, relevance_matrix)

        cls_beyond = relevance_matrix[0, 1:]
        normalized_scores = cls_beyond / (cls_beyond.max() + 1e-8)
        return normalized_scores.detach()


class ExplainabilityPipeline:
    def __init__(self, image_path: str = "dog.jpeg") -> None:
        self.image_path = image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Source image not found: {image_path}")

        self.pil_image = Image.open(image_path).convert("RGB")
        self.orig_width, self.orig_height = self.pil_image.size

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.input_tensor = self.transform(self.pil_image).unsqueeze(0)
        self.wrapper = ViTModelWrapper()

        with torch.no_grad():
            preds = self.wrapper.model(self.input_tensor)
            self.target_class = int(preds.argmax().item())
            self.predicted_label = self.wrapper.categories[self.target_class]
            self.confidence = float(F.softmax(preds, dim=-1)[0, self.target_class].item())

        self.algorithms: Dict[str, IExplanationAlgorithm] = {
            "Raw Attention": RawAttentionAlgorithm(),
            "Attention Rollout": AttentionRolloutAlgorithm(),
            "ViT Grad-CAM": ViTGradCAMAlgorithm(),
            "Full LRP": FullLRPAlgorithm(),
            "Partial LRP": PartialLRPAlgorithm(),
            "Beyond Attention": BeyondAttentionAlgorithm(),
        }

    def compute_spatial_contrast(self, grid_14x14: np.ndarray) -> Tuple[float, float, float]:
        foreground_mask = np.zeros((14, 14), dtype=bool)
        foreground_mask[2:12, 2:12] = True
        foreground_mean = float(grid_14x14[foreground_mask].mean())
        background_mean = float(grid_14x14[~foreground_mask].mean())
        contrast_ratio = foreground_mean / (background_mean + 1e-8)
        return foreground_mean, background_mean, contrast_ratio

    def run_all_methods(self) -> List[VisualExplanationResult]:
        print(f"Source Image:       {self.image_path} ({self.orig_width}x{self.orig_height})")
        print(f"Predicted Class:    {self.predicted_label} (ID: {self.target_class})")
        print(f"Model Confidence:   {self.confidence * 100:.2f}%")
        print("Patch Grid Shape:   14x14 (196 total patches, 16x16 pixels each)")

        results: List[VisualExplanationResult] = []

        for name, algo in self.algorithms.items():
            attribution_1d = algo.compute_attribution(self.wrapper, self.input_tensor, self.target_class)
            grid_14x14 = attribution_1d.reshape(14, 14).cpu().numpy()
            fg, bg, ratio = self.compute_spatial_contrast(grid_14x14)
            is_class_specific = name not in ["Raw Attention", "Attention Rollout"]

            results.append(
                VisualExplanationResult(
                    method_name=name,
                    attribution_map=grid_14x14,
                    foreground_score=fg,
                    background_score=bg,
                    contrast_ratio=ratio,
                    class_specific=is_class_specific
                )
            )

            print(f"[{name:<18}] FG: {fg:.4f} | BG: {bg:.4f} | Contrast: {ratio:.2f}x | Class-Specific: {is_class_specific}")

        self.save_visualizations(results)
        return results

    def save_visualizations(self, results: List[VisualExplanationResult]) -> None:
        rgb_original = np.array(self.pil_image)
        h, w, _ = rgb_original.shape

        # // Generate combined 7-panel visualization figure
        fig, axes = plt.subplots(1, 7, figsize=(28, 4), dpi=200)

        # // Panel 1: Original Image
        axes[0].imshow(rgb_original)
        axes[0].set_title(f"Input: {self.predicted_label}\n({self.confidence*100:.1f}%)", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        filename_map = {
            "Raw Attention": "output_raw_attention.png",
            "Attention Rollout": "output_attention_rollout.png",
            "ViT Grad-CAM": "output_vit_gradcam.png",
            "Full LRP": "output_full_lrp.png",
            "Partial LRP": "output_partial_lrp.png",
            "Beyond Attention": "output_beyond_attention.png",
        }

        for idx, res in enumerate(results, start=1):
            # // Upsample 14x14 grid to original image dimensions using bicubic interpolation
            tensor_grid = torch.tensor(res.attribution_map).unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(tensor_grid, size=(h, w), mode="bicubic", align_corners=False).squeeze().numpy()
            upsampled = np.clip(upsampled, 0.0, 1.0)

            # // Plot on multi-panel figure
            axes[idx].imshow(rgb_original)
            im = axes[idx].imshow(upsampled, cmap="jet", alpha=0.55)
            spec_str = "Class-Specific" if res.class_specific else "Class-Agnostic"
            axes[idx].set_title(f"{res.method_name}\n({spec_str})\nContrast: {res.contrast_ratio:.2f}x", fontsize=9)
            axes[idx].axis("off")

            # // Save individual standalone figure
            fig_single, ax_single = plt.subplots(figsize=(6, 5), dpi=150)
            ax_single.imshow(rgb_original)
            single_im = ax_single.imshow(upsampled, cmap="jet", alpha=0.55)
            ax_single.set_title(f"{res.method_name} - {self.predicted_label}", fontsize=11, fontweight="bold")
            ax_single.axis("off")
            plt.colorbar(single_im, ax=ax_single, fraction=0.046, pad=0.04)
            plt.tight_layout()
            out_path = filename_map[res.method_name]
            fig_single.savefig(out_path, bbox_inches="tight")
            plt.close(fig_single)

        plt.tight_layout()
        comparison_path = "output_xai_comparison.png"
        fig.savefig(comparison_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\nAll visualization images saved successfully:")
        print(f"Master comparison grid: {comparison_path}")
        for path in filename_map.values():
            print(f"Individual heatmap:     {path}")


def main() -> None:
    pipeline = ExplainabilityPipeline("dog.jpeg")
    pipeline.run_all_methods()


if __name__ == "__main__":
    main()
