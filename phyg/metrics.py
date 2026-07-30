"""Author-compatible 8-bit PSNR, SSIM and AlexNet LPIPS metrics."""

import cv2
import lpips
import numpy as np
import torch


def tensor_to_uint8_rgb(image):
    """Match torchvision ToPILImage float conversion: clamp, *255, uint8 truncation."""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("expected one image when input is NCHW")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("expected CHW RGB tensor")
    return (
        image.detach().clamp(0, 1).mul(255).to(torch.uint8)
        .permute(1, 2, 0).cpu().numpy()
    )


def calculate_psnr_uint8(target, reference):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(reference, dtype=np.float32)
    diff = img1 - img2
    return float(10.0 * np.log10(
        255.0 * 255.0 / (np.mean(np.square(diff)) + 1e-8)
    ))


def _ssim_channel(prediction, target):
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    score = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return score.mean()


def calculate_ssim_uint8(target, reference):
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(reference, dtype=np.float64)
    if img1.shape != img2.shape:
        raise ValueError("input images must have identical dimensions")
    if img1.ndim == 2:
        return float(_ssim_channel(img1, img2))
    if img1.ndim == 3 and img1.shape[2] == 3:
        return float(np.mean([
            _ssim_channel(img1[:, :, channel], img2[:, :, channel])
            for channel in range(3)
        ]))
    if img1.ndim == 3 and img1.shape[2] == 1:
        return float(_ssim_channel(np.squeeze(img1), np.squeeze(img2)))
    raise ValueError("wrong input image dimensions")


class AuthorValidationMetrics:
    def __init__(self, device):
        self.device = torch.device(device)
        self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()

    @torch.no_grad()
    def evaluate_uint8_pair(self, output_uint8, gt_uint8):
        output_uint8 = np.asarray(output_uint8, dtype=np.uint8)
        gt_uint8 = np.asarray(gt_uint8, dtype=np.uint8)
        output_tensor = lpips.im2tensor(output_uint8).to(self.device)
        gt_tensor = lpips.im2tensor(gt_uint8).to(self.device)
        return {
            "psnr": calculate_psnr_uint8(output_uint8, gt_uint8),
            "ssim": calculate_ssim_uint8(output_uint8, gt_uint8),
            "lpips": float(self.lpips_model.forward(gt_tensor, output_tensor).item()),
        }

    @torch.no_grad()
    def evaluate_tensor_pair(self, output, gt):
        return self.evaluate_uint8_pair(
            tensor_to_uint8_rgb(output), tensor_to_uint8_rgb(gt)
        )
