import argparse
import logging
import math
import os
import shutil
import json
from glob import glob
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from packaging import version
from functools import partial
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
)

from echo.common.datasets import TensorSetv5, ImageSet, TensorSetv6
from echo.common import (
    pad_reshape,
    unpad_reshape,
    padf,
    unpadf,
    load_model,
    save_as_mp4,
    save_as_gif,
    save_as_img,
    save_as_avi,
    parse_formats,
)

"""
CUDA_VISIBLE_DEVICES='4' python -m echo.lvdm.sample_multi_ref  
	--config echo/lvdm/configs/cardiac_asd.yaml   
	--unet /nfs/usrhome/khmuhammad/EchoPath/experiments/cardiac_asd/checkpoint-60000/unet_ema   
	--vae /nfs/usrhome/khmuhammad/EchoPath/models/vae   
	--conditioning /nfs/usrhome/khmuhammad/EchoPath/data/latents/cardiac_asd/Latents  
	--output /nfs/usrhome/khmuhammad/EchoPath/samples/lvdm_cardiac_asd_multi_ref  
	--num_samples 200    
	--batch_size 48    
	--num_steps 256     
	--save_as mp4,jpg    
	--frames 192 
	--sampling_mode diffusion 
	--conditioning_type class_id 
	--class_ids 2
    --condition_guidance_scale 5.0
    --seed 42
    --num_ref_frames 3
"""


def get_conditioning_vector(
    conditioning_type, conditioning_value, B, device, dtype, generator=None
):
    """
    Create conditioning vectors based on the specified type
    """
    if conditioning_type == "class_id":
        # Integer class IDs
        if isinstance(conditioning_value, int):
            # Random class IDs up to conditioning_value
            cond = torch.randint(
                0, conditioning_value, (B,), device=device, dtype=dtype
            )
        else:
            # Fixed class ID
            cond = torch.tensor(
                [int(conditioning_value)] * B, device=device, dtype=dtype
            )

        # Format for model: B -> B x 1 x 1
        return cond[:, None, None]

    elif conditioning_type == "lvef":
        # LVEF values (usually between 0-100)
        if (
            isinstance(conditioning_value, (list, tuple))
            and len(conditioning_value) == 2
        ):
            # Random LVEF in range
            min_val, max_val = conditioning_value
            cond = torch.randint(min_val, max_val+1, (B,), device=device, dtype=dtype, generator=generator)
            cond = cond / 100.0
        else:
            # Fixed LVEF value
            cond = torch.tensor(
                [float(conditioning_value)] * B, device=device, dtype=dtype
            )
            cond = cond / 100.0

        # Format for model: B -> B x 1 x 1
        return cond[:, None, None]

    elif conditioning_type == "view":
        # View type as integer ID
        if isinstance(conditioning_value, int):
            # Random view IDs up to conditioning_value
            cond = torch.randint(
                0, conditioning_value, (B,), device=device, dtype=dtype
            )
        else:
            # Fixed view ID
            cond = torch.tensor(
                [int(conditioning_value)] * B, device=device, dtype=dtype
            )

        # Format for model: B -> B x 1 x 1
        return cond[:, None, None]

    else:
        raise ValueError(f"Unsupported conditioning type: {conditioning_type}")


def create_multi_ref_frames(ref_frames_multi, T):
    """
    Create reference frames for the entire temporal sequence from multiple reference frames.

    Args:
        ref_frames_multi: Tensor of shape [B, C, num_ref_frames, H, W]
        T: Total number of frames in the sequence

    Returns:
        ref_frames_expanded: Tensor of shape [B, C, T, H, W]
    """
    B, C, num_ref_frames, H, W = ref_frames_multi.shape
    device = ref_frames_multi.device
    dtype = ref_frames_multi.dtype

    ref_frames_expanded = torch.zeros(B, C, T, H, W, device=device, dtype=dtype)

    if num_ref_frames == 1:
        # Single reference frame - replicate across all time steps
        ref_frames_expanded = ref_frames_multi[:, :, 0:1, :, :].repeat(1, 1, T, 1, 1)
    elif num_ref_frames == 3:
        # Three reference frames - apply to different segments
        # 0th frame for indices 0-20
        ref_frames_expanded[:, :, :21, :, :] = ref_frames_multi[:, :, 0:1, :, :].repeat(
            1, 1, min(21, T), 1, 1
        )

        if T > 21:
            # 32nd frame equivalent for indices 21-42
            ref_frames_expanded[:, :, 21:43, :, :] = ref_frames_multi[
                :, :, 1:2, :, :
            ].repeat(1, 1, min(22, T - 21), 1, 1)

        if T > 43:
            # 63rd frame equivalent for indices 43-end
            ref_frames_expanded[:, :, 43:, :, :] = ref_frames_multi[
                :, :, 2:3, :, :
            ].repeat(1, 1, T - 43, 1, 1)
    else:
        # For other numbers of reference frames, distribute evenly
        segment_length = T // num_ref_frames
        remainder = T % num_ref_frames

        start_idx = 0
        for i in range(num_ref_frames):
            # Calculate segment size (add 1 to first 'remainder' segments)
            current_segment_length = segment_length + (1 if i < remainder else 0)
            end_idx = start_idx + current_segment_length

            if start_idx < T:
                ref_frames_expanded[:, :, start_idx : min(end_idx, T), :, :] = (
                    ref_frames_multi[:, :, i : i + 1, :, :].repeat(
                        1, 1, min(current_segment_length, T - start_idx), 1, 1
                    )
                )

            start_idx = end_idx
    return ref_frames_expanded


if __name__ == "__main__":
    # 1 - Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--unet", type=str, default=None, help="Path unet checkpoint.")
    parser.add_argument("--vae", type=str, default=None, help="Path vae checkpoint.")
    parser.add_argument(
        "--conditioning",
        type=str,
        default=None,
        help="Path to the folder containing the conditioning latents/images.",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory.")
    parser.add_argument(
        "--num_samples", type=int, default=8, help="Number of samples to generate."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of steps.")

    # New arguments for flexible conditioning and sampling
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="class_id",
        choices=["class_id", "lvef", "view", "csv"],
        help="Type of conditioning to use.",
    )

    # Multi-reference specific arguments
    parser.add_argument(
        "--num_ref_frames",
        type=int,
        default=3,
        help="Number of reference frames to use for conditioning.",
    )

    # Conditioning value arguments - one will be used based on conditioning_type
    parser.add_argument(
        "--class_ids",
        type=int,
        default=2,
        help="Number of class ids or specific class id.",
    )
    parser.add_argument(
        "--lvef_range",
        type=float,
        nargs=2,
        default=[10, 90],
        help="Min and max LVEF values.",
    )
    parser.add_argument("--lvef", type=float, default=None, help="Specific LVEF value.")
    parser.add_argument(
        "--view_ids",
        type=int,
        default=4,
        help="Number of view ids or specific view id.",
    )

    parser.add_argument(
        "--save_as",
        type=parse_formats,
        default=None,
        help="Save formats separated by commas (e.g., avi,jpg). Available: avi, mp4, gif, jpg, png, pt",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=64,
        help="Number of frames to generate. Must be a multiple of 32",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--condition_guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for class conditioning (1.0=no guidance).",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="Use DDIM sampler.",
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # 2 - Load models
    unet = load_model(args.unet)
    vae = load_model(args.vae)

    # 3 - Load scheduler
    scheduler_kwargs = OmegaConf.to_container(config.noise_scheduler)
    scheduler_klass_name = scheduler_kwargs.pop("_class_name")
    if args.ddim:
        print("Using DDIMScheduler")
        scheduler_klass_name = "DDIMScheduler"
        scheduler_kwargs.pop("variance_type", None)
    scheduler_klass = getattr(diffusers, scheduler_klass_name, None)
    assert scheduler_klass is not None, (
        f"Could not find scheduler class {scheduler_klass_name}"
    )
    scheduler = scheduler_klass(**scheduler_kwargs)

    scheduler.set_timesteps(args.num_steps)
    timesteps = scheduler.timesteps

    # 4 - Load dataset for reference frames with multiple frames support
    file_ext = os.listdir(args.conditioning)[0].split(".")[-1].lower()
    assert file_ext in ["pt", "jpg", "png"], (
        f"Conditioning files must be either .pt, .jpg or .png, not {file_ext}"
    )

    if file_ext == "pt":
        # dataset = TensorSetv5(args.conditioning) # Uncomment for cardiac_asd
        dataset = TensorSetv6(args.conditioning)
    else:
        # For image files, we'll need to modify to handle multiple frames
        # For now, use single frame and replicate
        dataset = ImageSet(args.conditioning, ext=file_ext)
        args.num_ref_frames = 1  # Override to 1 for image inputs

    assert len(dataset) > 0, (
        f"No files found in {args.conditioning} with extension {file_ext}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True,
    )

    # 5 - Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(
        args.seed
        if args.seed is not None
        else config.seed
        if hasattr(config, "seed")
        else np.random.randint(0, 1000000)
    )
    unet = unet.to(device, dtype)
    vae = vae.to(device, torch.float32)
    unet.eval()
    vae.eval()

    format_input = (
        pad_reshape
        if config.unet._class_name == "UNetSpatioTemporalConditionModel"
        else padf
    )
    format_output = (
        unpad_reshape
        if config.unet._class_name == "UNetSpatioTemporalConditionModel"
        else unpadf
    )

    B, C, T, H, W = (
        args.batch_size,
        config.unet.out_channels,
        config.unet.num_frames,
        config.unet.sample_size,
        config.unet.sample_size,
    )
    fps = config.globals.target_fps if hasattr(config.globals, "target_fps") else 30

    # no stitching anymore - single chunk processing
    args.frames = min(args.frames, T)  # Cap frames to model's max capacity

    # Forward kwargs setup
    forward_kwargs = {
        "timestep": -1,  # Will be updated in the loop
    }

    # Set up conditioning based on type
    if args.conditioning_type == "lvef" and args.lvef is not None:
        conditioning_value = args.lvef
    elif args.conditioning_type == "lvef":
        conditioning_value = args.lvef_range
    elif args.conditioning_type == "class_id":
        conditioning_value = args.class_ids
    elif args.conditioning_type == "view":
        conditioning_value = args.view_ids
    else:
        conditioning_value = None  # For csv we'll handle differently

    if config.unet._class_name == "UNetSpatioTemporalConditionModel":
        dummy_added_time_ids = torch.zeros(
            (B, config.unet.addition_time_embed_dim), device=device, dtype=dtype
        )
        forward_kwargs["added_time_ids"] = dummy_added_time_ids

    sample_index = 0
    filelist = []

    os.makedirs(args.output, exist_ok=True)
    for ext in args.save_as:
        os.makedirs(os.path.join(args.output, ext), exist_ok=True)
    finished = False

    pbar = tqdm(total=args.num_samples)

    # 6 - Generate samples
    with torch.no_grad():
        while not finished:
            # for  cond, value in dataloader: # Uncomment for cardiac_asd with csv conditioning
            for cond in dataloader:
                if finished:
                    break

                # Prepare latent noise
                latents = torch.randn(
                    (B, C, T, H, W), device=device, dtype=dtype, generator=generator
                )

                # Get conditioning based on specified type
                if args.conditioning_type == "csv": 
                    print("Loading conditioning from CSV")
                    # conditioning = value[:, None, None] # Uncomment if you want to support CSV conditioning
                    # conditioning = conditioning.to(device, dtype=dtype)
                else:
                    # Get conditioning vector based on type
                    # This will handle class_id, lvef, view, etc.
                    conditioning = get_conditioning_vector(
                        args.conditioning_type,
                        conditioning_value,
                        B,
                        device,
                        dtype,
                        generator
                    )

                # Set the correct keyword argument based on conditioning type
                forward_kwargs["encoder_hidden_states"] = conditioning

                # Prepare reference frames
                latent_cond_images = cond.to(device, torch.float32)
                if latent_cond_images.dim() == 4:
                    print(f"You are probably using the sampled triplets, got {latent_cond_images.shape}")
                    _, CT, _, _ = latent_cond_images.shape
                    LC = 4  # Latent channels
                    LT = CT // LC

                    assert CT % LC == 0, "C*T dimension must be divisible by C"

                    # Split into T frames, each of shape (B, C, H, W)
                    frames = [
                        latent_cond_images[:, i * LC : (i + 1) * LC, :, :]
                        for i in range(LT)
                    ]  # list of T tensors

                    # Stack across time (dim=2) to get shape: (B, C, T, H, W)
                    latent_cond_images = torch.stack(frames, dim=2)
                    n = latent_cond_images.shape[2]
                    # Select the same random frame for all batch elements
                    random_frame_idx = torch.randint(0, n, (1,)).item()
                    latent_cond_images = latent_cond_images[:, :, random_frame_idx:random_frame_idx+1, :, :]  # Shape: (B, C, 1, H, W)
                    # print(
                    #     f"Conditioning shape after random selection: {latent_cond_images.shape}, dtype: {latent_cond_images.dtype}"
                    # )
                    
                elif latent_cond_images.dim() == 5:
                    print(
                        "You are probably using the random latents from the original test set"
                    )
                    # permute from B x num_ref_frames x C x H x W to B x C x num_ref_frames x H x W
                    latent_cond_images = latent_cond_images.permute(
                        0, 2, 1, 3, 4
                    )  # B x C x num_ref_frames x H x W

                else:
                    raise ValueError(
                        f"Conditioning latents must be 4D or 5D tensor, got {latent_cond_images.dim()}D"
                    )
                # print(
                #     f"Conditioning shape before expansion: {latent_cond_images.shape}, dtype: {latent_cond_images.dtype}"
                # )
                # Create expanded reference frames for the entire temporal sequence
                ref_frames_expanded = create_multi_ref_frames(latent_cond_images, T)
                # print(
                #     f"Conditioning shape after expansion: {ref_frames_expanded.shape}, dtype: {ref_frames_expanded.dtype}"
                # )

                # Apply classifier-free guidance if specified
                use_condition_guidance = args.condition_guidance_scale > 1.0

                # Prepare conditioning for CFG
                if use_condition_guidance:
                    # Create null conditioning
                    uncond_conditioning = torch.zeros_like(conditioning)
                    # Concatenate [unconditional, conditional]
                    conditioning = torch.cat([uncond_conditioning, conditioning])
                    # Duplicate reference frames
                    ref_frames_expanded = torch.cat([ref_frames_expanded] * 2)

                # Set up forward kwargs
                forward_kwargs["encoder_hidden_states"] = conditioning
                
                # Handle added_time_ids for CFG
                if config.unet._class_name == "UNetSpatioTemporalConditionModel":
                    if use_condition_guidance:
                        forward_kwargs["added_time_ids"] = torch.cat([dummy_added_time_ids] * 2)
                    else:
                        forward_kwargs["added_time_ids"] = dummy_added_time_ids

                # Denoise the latent
                with torch.autocast("cuda"):
                    # Diffusion sampling loop
                    for t in timesteps:
                        latent_model_input = (
                            torch.cat([latents] * 2)
                            if use_condition_guidance
                            else latents
                        )
                        latent_model_input = scheduler.scale_model_input(
                            latent_model_input, timestep=t
                        )
                        latent_model_input = torch.cat(
                            (latent_model_input, ref_frames_expanded), dim=1
                        )
                        latent_model_input, padding = format_input(latent_model_input, mult=3)

                        forward_kwargs["timestep"] = t
                        # forward_kwargs["encoder_hidden_states"] = conditioning

                        noise_pred = unet(latent_model_input, **forward_kwargs).sample
                        noise_pred = format_output(noise_pred, pad=padding)

                        if use_condition_guidance:
                            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + args.condition_guidance_scale * (
                                noise_pred_cond - noise_pred_uncond
                            )

                        latents = scheduler.step(noise_pred, t, latents).prev_sample

                # VAE decode
                latents = rearrange(latents, "b c t h w -> (b t) c h w").cpu()
                latents = latents / vae.config.scaling_factor

                # Decode in chunks to save memory
                chunked_latents = torch.split(latents, args.batch_size, dim=0)
                decoded_chunks = []
                for chunk in chunked_latents:
                    decoded_chunks.append(vae.decode(chunk.float().cuda()).sample.cpu())
                video = torch.cat(decoded_chunks, dim=0)  # (B*T) x H x W x C

                # format output
                video = rearrange(video, "(b t) c h w -> b t h w c", b=B)
                video = (video + 1) * 128
                video = video.clamp(0, 255).to(torch.uint8)

                print(
                    f"Generated videos: {video.shape}, dtype: {video.dtype}, range: [{video.min()}, {video.max()}]"
                )

                # Get conditioning values for metadata
                if args.conditioning_type == "class_id":
                    cond_values = conditioning.squeeze().to(torch.int).tolist()
                elif args.conditioning_type == "lvef":
                    cond_values = conditioning.squeeze().tolist()
                elif args.conditioning_type == "view":
                    cond_values = conditioning.squeeze().to(torch.int).tolist()
                elif args.conditioning_type == "csv":
                    cond_values = conditioning.squeeze().to(torch.int).tolist()

                # save samples
                for j in range(B):
                    # FileName,CondType,CondValue,FrameHeight,FrameWidth,FPS,NumberOfFrames,Split,NumRefFrames
                    filelist.append(
                        [
                            f"sample_{sample_index:06d}",
                            args.conditioning_type,
                            cond_values[j],
                            video.shape[2],
                            video.shape[3],
                            fps,
                            video.shape[1],
                            "GENERATED",
                            args.num_ref_frames,
                        ]
                    )

                    # Save in requested formats
                    if "mp4" in args.save_as:
                        save_as_mp4(
                            video[j],
                            os.path.join(
                                args.output, "mp4", f"sample_{sample_index:06d}.mp4"
                            ),
                        )
                    if "avi" in args.save_as:
                        save_as_avi(
                            video[j],
                            os.path.join(
                                args.output, "avi", f"sample_{sample_index:06d}.avi"
                            ),
                        )
                    if "gif" in args.save_as:
                        save_as_gif(
                            video[j],
                            os.path.join(
                                args.output, "gif", f"sample_{sample_index:06d}.gif"
                            ),
                        )
                    if "jpg" in args.save_as:
                        save_as_img(
                            video[j],
                            os.path.join(
                                args.output, "jpg", f"sample_{sample_index:06d}"
                            ),
                            ext="jpg",
                        )
                    if "png" in args.save_as:
                        save_as_img(
                            video[j],
                            os.path.join(
                                args.output, "png", f"sample_{sample_index:06d}"
                            ),
                            ext="png",
                        )
                    if "pt" in args.save_as:
                        torch.save(
                            video[j].clone(),
                            os.path.join(
                                args.output, "pt", f"sample_{sample_index:06d}.pt"
                            ),
                        )

                    sample_index += 1
                    pbar.update(1)
                    if sample_index >= args.num_samples:
                        finished = True
                        break

    # Save metadata
    df = pd.DataFrame(
        filelist,
        columns=[
            "FileName",
            "CondType",
            "CondValue",
            "FrameHeight",
            "FrameWidth",
            "FPS",
            "NumberOfFrames",
            "Split",
            "NumRefFrames",
        ],
    )
    df.to_csv(os.path.join(args.output, "FileList.csv"), index=False)

    # Save generation parameters
    params = {
        "conditioning_type": args.conditioning_type,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "condition_guidance_scale": args.condition_guidance_scale,
        "seed": args.seed,
        "frames": args.frames,
        "num_ref_frames": args.num_ref_frames,
    }
    with open(os.path.join(args.output, "generation_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(
        f"Generated {sample_index} samples using diffusion with {args.conditioning_type} conditioning and {args.num_ref_frames} reference frames."
    )
