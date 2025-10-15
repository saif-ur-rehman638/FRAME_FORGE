#!/usr/bin/env python3
"""
FrameForge The Final - Unified Interpolation + Multi-ControlNet Stylization
===========================================================================

This single script generates in-between frames between two input images using
ECCV2022-RIFE for interpolation and stylizes every interpolated frame with
Stable Diffusion Anything V5 guided by multiple ControlNets simultaneously
(Canny, LineArt, OpenPose). It aims to be the one-stop, production-ready
entrypoint for this project.

Inputs:
- data/start_frame.png
- data/end_frame.png
- A text prompt describing the transition/motion or style

Outputs:
- output_frames/frame_XXX.png

Models:
- RIFE (ECCV2022-RIFE) for frame interpolation
- Anything-V5 (local) for base SD model
- ControlNet Canny/LineArt/OpenPose (local)
- VAE sd-vae-ft-mse (local)
"""

import os
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ==================== CONFIGURATION ====================

# Paths (relative to this script directory)
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
START_FRAME = DATA_DIR / "start_frame.png"
END_FRAME = DATA_DIR / "end_frame.png"
OUTPUT_DIR = SCRIPT_DIR / "output_frames"
TEMP_DIR = SCRIPT_DIR / "temp_processing"
RIFE_INPUT_DIR = SCRIPT_DIR / "rife_input"
RIFE_OUTPUT_DIR = SCRIPT_DIR / "rife_output"

# RIFE
RIFE_SCRIPT = SCRIPT_DIR / "ECCV2022-RIFE" / "inference_img.py"
RIFE_MODEL_DIR = SCRIPT_DIR / "ECCV2022-RIFE" / "train_log"

# Stable Diffusion + ControlNet local model paths
BASE_MODEL = SCRIPT_DIR / "models" / "anything-v5"
VAE_PATH = SCRIPT_DIR / "models" / "vae" / "sd-vae-ft-mse"
CONTROLNET_DIRS = {
    "canny": SCRIPT_DIR / "models" / "controlnet" / "control_v11p_sd15_canny",
    "lineart": SCRIPT_DIR / "models" / "controlnet" / "control_v11p_sd15_lineart",
    "openpose": SCRIPT_DIR / "models" / "controlnet" / "control_v11p_sd15_openpose",
}


# Defaults
DEFAULT_NUM_FRAMES = 16  # must be power-of-two (2^n) for RIFE exp steps
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE = 7.5
DEFAULT_SEED = 42
DEFAULT_CONTROL_TYPES = ["canny", "lineart", "openpose"]  # use all by default

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== UTILITIES ====================

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RIFE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RIFE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs():
    if not START_FRAME.exists():
        print(f"[!] Start frame not found: {START_FRAME}")
        return False
    if not END_FRAME.exists():
        print(f"[!] End frame not found: {END_FRAME}")
        return False
    if not RIFE_SCRIPT.exists():
        print(f"[!] RIFE script not found: {RIFE_SCRIPT}")
        return False
    if not RIFE_MODEL_DIR.exists():
        print(f"[!] RIFE model dir not found: {RIFE_MODEL_DIR}")
        return False
    if not BASE_MODEL.exists():
        print(f"[!] Base model not found: {BASE_MODEL}")
        return False
    for k, p in CONTROLNET_DIRS.items():
        if not p.exists():
            print(f"[!] ControlNet '{k}' missing at {p}")
            return False
    if not VAE_PATH.exists():
        print(f"[!] VAE path missing at {VAE_PATH}")
        return False
    return True


def make_divisible_by_8(w: int, h: int) -> tuple[int, int]:
    return ((w + 7) // 8) * 8, ((h + 7) // 8) * 8


def resize_and_pad(image: Image.Image, size_wh: tuple[int, int]) -> Image.Image:
    scale = min(size_wh[0] / image.width, size_wh[1] / image.height)
    new_size = (int(image.width * scale), int(image.height * scale))
    resized = image.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", size_wh, (0, 0, 0))
    dx = (size_wh[0] - new_size[0]) // 2
    dy = (size_wh[1] - new_size[1]) // 2
    canvas.paste(resized, (dx, dy))
    return canvas


# ==================== RIFE INTERPOLATION ====================

def prepare_rife_inputs() -> tuple[int, int]:
    print("[+] Preparing frames for RIFE...")
    img0 = Image.open(START_FRAME).convert("RGB")
    img1 = Image.open(END_FRAME).convert("RGB")
    max_w = max(img0.width, img1.width)
    max_h = max(img0.height, img1.height)
    max_w, max_h = make_divisible_by_8(max_w, max_h)
    img0 = resize_and_pad(img0, (max_w, max_h))
    img1 = resize_and_pad(img1, (max_w, max_h))
    (RIFE_INPUT_DIR / "0.png").parent.mkdir(parents=True, exist_ok=True)
    img0.save(RIFE_INPUT_DIR / "0.png")
    img1.save(RIFE_INPUT_DIR / "1.png")
    print(f"[+] RIFE input size: {max_w}x{max_h}")
    return max_w, max_h


def run_rife_interpolation(num_frames: int) -> bool:
    print("[+] Running RIFE interpolation...")
    # num_frames must be 2^n
    exp_steps = int(np.log2(num_frames))

    if RIFE_OUTPUT_DIR.exists():
        shutil.rmtree(RIFE_OUTPUT_DIR)
    RIFE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RIFE_SCRIPT),
        "--img",
        str(RIFE_INPUT_DIR / "0.png"),
        str(RIFE_INPUT_DIR / "1.png"),
        "--exp",
        str(exp_steps),
        "--model",
        str(RIFE_MODEL_DIR),
    ]

    import subprocess
    print("[+] Exec:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[!] RIFE failed")
        print(result.stdout)
        print(result.stderr)
        # Fallback: synthesize crossfaded in-between frames
        try:
            print("[!] Falling back to crossfade interpolation...")
            img0 = Image.open(RIFE_INPUT_DIR / "0.png").convert("RGB")
            img1 = Image.open(RIFE_INPUT_DIR / "1.png").convert("RGB")
            total = num_frames
            for i in range(total):
                t = 0 if total <= 1 else i / (total - 1)
                blended = Image.blend(img0, img1, t)
                blended.save(RIFE_OUTPUT_DIR / f"frame_{i:03d}.png")
            print(f"[+] Crossfaded {total} frames as fallback")
            return True
        except Exception as e:
            print(f"[!] Crossfade fallback failed: {e}")
            return False

    # RIFE writes to ./output/*.png; move into RIFE_OUTPUT_DIR as frame_XXX.png
    out_glob = list(sorted(glob.glob(str(SCRIPT_DIR / "output" / "*.png"))))
    for i, p in enumerate(out_glob):
        shutil.move(p, RIFE_OUTPUT_DIR / f"frame_{i:03d}.png")
    out_dir = SCRIPT_DIR / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"[+] RIFE generated {len(list(RIFE_OUTPUT_DIR.glob('*.png')))} frames")
    return True


# ==================== CONTROLNET STYLIZATION ====================

def load_pipelines(control_types: list[str]):
    print(f"[+] Loading SD Anything-V5 + ControlNet(s): {control_types}")
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        AutoencoderKL,
    )

    control_models = []
    for ct in control_types:
        control_models.append(
            ControlNetModel.from_pretrained(
                str(CONTROLNET_DIRS[ct]), torch_dtype=torch.float16
            )
        )

    vae = AutoencoderKL.from_pretrained(str(VAE_PATH), torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        str(BASE_MODEL),
        controlnet=control_models,
        vae=vae,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(DEVICE)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[+] xFormers attention enabled")
    except Exception:
        print("[!] xFormers not available; continuing without it")

    return pipe


def create_control_images(image: Image.Image, control_types: list[str]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for ct in control_types:
        if ct == "canny":
            import cv2
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            edges = cv2.Canny(img_cv, 100, 200)
            images.append(Image.fromarray(edges).convert("RGB"))
        elif ct == "lineart":
            try:
                from controlnet_aux import LineartDetector
                detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
                images.append(detector(image))
            except Exception:
                # Fallback to simple edges
                import cv2
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                edges = cv2.Canny(img_cv, 100, 200)
                images.append(Image.fromarray(edges).convert("RGB"))
        elif ct == "openpose":
            try:
                from controlnet_aux import OpenposeDetector
                detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                images.append(detector(image))
            except Exception:
                import cv2
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                edges = cv2.Canny(img_cv, 100, 200)
                images.append(Image.fromarray(edges).convert("RGB"))
        else:
            raise ValueError(f"Unknown control type: {ct}")
    return images


def stylize_frames(
    prompt: str,
    control_types: list[str],
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
):
    print("[+] Stylizing interpolated frames...")
    from torch import Generator

    pipe = load_pipelines(control_types)
    generator = Generator(device=DEVICE).manual_seed(seed)

    frame_paths = list(sorted(RIFE_OUTPUT_DIR.glob("*.png")))
    if not frame_paths:
        print("[!] No interpolated frames found")
        return False

    for i, fp in enumerate(tqdm(frame_paths, desc="Stylizing")):
        frame = Image.open(fp).convert("RGB")
        control_images = create_control_images(frame, control_types)
        # Resize control images to match frame in case detectors returned different sizes
        control_images = [ci.resize(frame.size, Image.NEAREST) for ci in control_images]

        try:
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    image=control_images,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=frame.height,
                    width=frame.width,
                ).images[0]
        except Exception as e:
            print(f"[!] Stylization failed at frame {i}: {e}")
            result = frame  # fallback

        out_path = OUTPUT_DIR / f"frame_{i:03d}.png"
        result.save(out_path)

    print(f"[+] Stylized frames saved to {OUTPUT_DIR}")
    return True


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(
        description="FrameForge The Final: Interpolate with RIFE, stylize with SD + multi-ControlNet"
    )
    parser.add_argument("--description", "-d", required=True, help="Text prompt")
    parser.add_argument(
        "--num-frames",
        "-n",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Total frames including endpoints; must be power-of-two",
    )
    parser.add_argument(
        "--control",
        "-c",
        nargs="+",
        default=DEFAULT_CONTROL_TYPES,
        choices=["canny", "lineart", "openpose", "all"],
        help="One or more control types; use 'all' for canny lineart openpose",
    )
    parser.add_argument(
        "--steps", "-i", type=int, default=DEFAULT_STEPS, help="Diffusion steps"
    )
    parser.add_argument(
        "--guidance", "-g", type=float, default=DEFAULT_GUIDANCE, help="CFG scale"
    )
    parser.add_argument("--seed", "-s", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep rife_input/rife_output directories after completion",
    )

    args = parser.parse_args()

    # Normalize control selection
    control_types = args.control
    if "all" in control_types:
        control_types = DEFAULT_CONTROL_TYPES

    # Validate frames count is power-of-two
    if abs(np.log2(args.num_frames) - int(np.log2(args.num_frames))) > 1e-6:
        print("[!] --num-frames must be a power of two (e.g., 8, 16, 32)")
        sys.exit(1)

    print("[*] Starting FrameForge The Final...")
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Prompt: {args.description}")
    print(f"[*] ControlNets: {control_types}")
    print(f"[*] Frames: {args.num_frames}")
    print("-" * 50)

    ensure_dirs()
    if not validate_inputs():
        sys.exit(1)

    prepare_rife_inputs()
    if not run_rife_interpolation(args.num_frames):
        sys.exit(1)

    if not stylize_frames(
        prompt=args.description,
        control_types=control_types,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
    ):
        sys.exit(1)

    if not args.keep_intermediate:
        if RIFE_INPUT_DIR.exists():
            shutil.rmtree(RIFE_INPUT_DIR)
        if RIFE_OUTPUT_DIR.exists():
            shutil.rmtree(RIFE_OUTPUT_DIR)

    print("-" * 50)
    print("[+] Completed successfully!")
    print(f"[+] Frames at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


