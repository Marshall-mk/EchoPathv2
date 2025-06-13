#!/usr/bin/env python
"""
Compute MSE, MAE, SSIM, PSNR and LPIPS between two folders of images
that have 1-to-1 filename correspondence.

Example:
    python compute_recon_metrics.py \
        --ref_dir  path/to/originals \
        --pred_dir path/to/reconstructed \
        --ext      .jpg \
        --output_dir path/to/save/metrics
"""

import argparse, os, glob, pathlib, sys, warnings
from collections import defaultdict
from datetime import datetime

import torch
from PIL import Image
from torchvision import transforms
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    LearnedPerceptualImagePatchSimilarity,
)

# --------------------------- helpers ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()  # converts PIL image to [0,1] float tensor

def load_img(path):
    """Return CHW float tensor in [0,1] range on current device."""
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

# --------------------------- main ------------------------------------
def main(args):
    ref_paths = sorted(glob.glob(os.path.join(args.ref_dir, f"*{args.ext}")))
    if len(ref_paths) == 0:
        sys.exit(f"No files with extension '{args.ext}' found in {args.ref_dir}")

    # match each reference file with its counterpart in pred_dir
    pairs = []
    for r in ref_paths:
        fname = pathlib.Path(r).name
        p = os.path.join(args.pred_dir, fname)
        if not os.path.isfile(p):
            warnings.warn(f"Missing prediction for {fname}; skipping")
            continue
        pairs.append((r, p))
    if len(pairs) == 0:
        sys.exit("No matching file pairs found.")

    # --- metric objects (torchmetrics keeps running statistics) -------
    mse   = MeanSquaredError().to(device)
    mae   = MeanAbsoluteError().to(device)
    ssim  = StructuralSimilarityIndexMeasure(data_range=1.).to(device)
    psnr  = PeakSignalNoiseRatio(data_range=1.).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    # We'll also store per-image values to compute std-dev later
    per_img = defaultdict(list)

    for ref_path, pred_path in pairs:
        ref  = load_img(ref_path)
        pred = load_img(pred_path)
        # update stateful metrics
        mse.update(pred, ref)
        mae.update(pred, ref)
        ssim.update(pred, ref)
        psnr.update(pred, ref)
        lpips.update(pred, ref)
        # record individual values for std
        with torch.no_grad():
            per_img["MSE" ].append(torch.nn.functional.mse_loss(pred, ref).item())
            per_img["MAE" ].append(torch.nn.functional.l1_loss (pred, ref).item())
            per_img["SSIM"].append(ssim(pred, ref).item())
            per_img["PSNR"].append(psnr(pred, ref).item())
            per_img["LPIPS"].append(lpips(pred, ref).item())

    # compute global means from torchmetrics
    results = {
        "MSE"  : mse.compute().item(),
        "MAE"  : mae.compute().item(),
        "SSIM" : ssim.compute().item(),
        "PSNR" : psnr.compute().item(),
        "LPIPS": lpips.compute().item(),
    }

    # add std-dev
    for k in results:
        std = torch.tensor(per_img[k]).std().item()
        results[k] = f"{results[k]:.4f} ±{std:.4f}"

    # ---- pretty print ------------------------------------------------
    try:
        import pandas as pd
        from tabulate import tabulate
        df = pd.DataFrame(results, index=["score"]).T
        table_str = tabulate(df, headers=["Metric", "Mean ± Std"], tablefmt="github")
        print(table_str)
    except ImportError:
        table_str = "\n".join(f"{k}: {v}" for k,v in results.items())
        print(table_str)

    # ---- save to file if output_dir specified ------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate filename based on directory names and timestamp
        ref_name = pathlib.Path(args.ref_dir).name
        pred_name = pathlib.Path(args.pred_dir).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{ref_name}_vs_{pred_name}_{timestamp}.txt"
        output_path = os.path.join(args.output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write(f"Reconstruction Metrics Comparison\n")
            f.write(f"Reference Directory: {args.ref_dir}\n")
            f.write(f"Prediction Directory: {args.pred_dir}\n")
            f.write(f"Image Extension: {args.ext}\n")
            f.write(f"Number of Image Pairs: {len(pairs)}\n")
            f.write(f"Computed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\n{'-'*50}\n\n")
            if 'table_str' in locals():
                f.write(table_str)
            else:
                f.write("\n".join(f"{k}: {v}" for k,v in results.items()))
        
        print(f"\nMetrics saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir",  required=True, help="Folder with ground-truth images")
    parser.add_argument("--pred_dir", required=True, help="Folder with reconstructed images")
    parser.add_argument("--ext", default=".png", help="Image extension (default: .png)")
    parser.add_argument("--output_dir", help="Directory to save metrics file (optional)")
    main(parser.parse_args())
