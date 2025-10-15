#!/usr/bin/env python3
"""
FrameForge Pipeline - AI-Powered Frame Interpolation and Stylization
====================================================================

This pipeline takes two input frames (start and end) and generates a sequence
of stylized intermediate frames using:
1. ECCV2022-RIFE for frame interpolation
2. Stable Diffusion + ControlNet for stylization

Input:
- start_frame.png: Starting frame
- end_frame.png: Ending frame  
- middle_description: Text description of what happens between frames

Output:
- Generated sequence of stylized frames (configurable count)

Models Used:
- ECCV2022-RIFE: Frame interpolation
- Anything-V5: Base Stable Diffusion model
- ControlNet: Canny/LineArt/OpenPose for structure control
- VAE: SD-VAE-FT-MSE for better quality
"""

import os
import sys
import subprocess
import shutil
import glob
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# ==================== CONFIGURATION ====================

# Input/Output paths
START_FRAME = "data/start_frame.png"
END_FRAME = "data/end_frame.png"
OUTPUT_DIR = "output_frames"
TEMP_DIR = "temp_processing"

# RIFE Configuration
RIFE_SCRIPT = "ECCV2022-RIFE/inference_img.py"
RIFE_MODEL_DIR = "ECCV2022-RIFE/train_log"
NUM_INTERPOLATED_FRAMES = 8  # Total frames to generate (including start/end)

# Stable Diffusion Configuration
BASE_MODEL = "./models/anything-v5"
VAE_PATH = "./models/vae/sd-vae-ft-mse"
CONTROLNET_PATHS = {
    "canny": "./models/controlnet/control_v11p_sd15_canny",
    "lineart": "./models/controlnet/control_v11p_sd15_lineart", 
    "openpose": "./models/controlnet/control_v11p_sd15_openpose"
}

# Default settings
DEFAULT_CONTROL_TYPE = "canny"
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 512

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== UTILITY FUNCTIONS ====================

def setup_directories():
    """Create necessary directories for processing."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs("rife_input", exist_ok=True)
    os.makedirs("rife_output", exist_ok=True)
    print(f"[+] Created directories: {OUTPUT_DIR}, {TEMP_DIR}")

def validate_inputs():
    """Validate that required input files exist."""
    if not os.path.exists(START_FRAME):
        print(f"[!] ERROR: Start frame not found at {START_FRAME}")
        return False
    if not os.path.exists(END_FRAME):
        print(f"[!] ERROR: End frame not found at {END_FRAME}")
        return False
    if not os.path.exists(RIFE_SCRIPT):
        print(f"[!] ERROR: RIFE script not found at {RIFE_SCRIPT}")
        return False
    if not os.path.exists(RIFE_MODEL_DIR):
        print(f"[!] ERROR: RIFE model directory not found at {RIFE_MODEL_DIR}")
        return False
    print("[+] Input validation passed")
    return True

def resize_and_pad_image(image, target_size):
    """Resize and pad image to target size while maintaining aspect ratio."""
    # Calculate scaling to fit within target size
    scale = min(target_size[0] / image.width, target_size[1] / image.height)
    new_size = (int(image.width * scale), int(image.height * scale))
    
    # Resize image
    resized = image.resize(new_size, Image.LANCZOS)
    
    # Create padded image
    padded = Image.new("RGB", target_size, (0, 0, 0))
    x_offset = (target_size[0] - new_size[0]) // 2
    y_offset = (target_size[1] - new_size[1]) // 2
    padded.paste(resized, (x_offset, y_offset))
    
    return padded

# ==================== RIFE INTERPOLATION ====================

def prepare_rife_input():
    """Prepare input frames for RIFE interpolation."""
    print("[+] Preparing RIFE input frames...")
    
    # Load and process start/end frames
    start_img = Image.open(START_FRAME).convert("RGB")
    end_img = Image.open(END_FRAME).convert("RGB")
    
    # Ensure both images are the same size (use the larger dimensions)
    max_width = max(start_img.width, end_img.width)
    max_height = max(start_img.height, end_img.height)
    
    # Make dimensions divisible by 8 (RIFE requirement)
    max_width = ((max_width + 7) // 8) * 8
    max_height = ((max_height + 7) // 8) * 8
    
    # Resize and pad both images
    start_processed = resize_and_pad_image(start_img, (max_width, max_height))
    end_processed = resize_and_pad_image(end_img, (max_width, max_height))
    
    # Save processed images
    start_processed.save("rife_input/0.png")
    end_processed.save("rife_input/1.png")
    
    print(f"[+] RIFE input prepared: {max_width}x{max_height}")
    return max_width, max_height

def run_rife_interpolation():
    """Run RIFE to generate intermediate frames."""
    print("[+] Running RIFE frame interpolation...")
    
    # Calculate number of interpolation steps needed
    # For 8 total frames, we need 3 interpolation steps (2^3 = 8)
    exp_steps = int(np.log2(NUM_INTERPOLATED_FRAMES))
    
    # Clear previous output
    if os.path.exists("rife_output"):
        shutil.rmtree("rife_output")
    os.makedirs("rife_output", exist_ok=True)
    
    # Run RIFE inference
    cmd = [
        sys.executable, RIFE_SCRIPT,
        "--img", "rife_input/0.png", "rife_input/1.png",
        "--exp", str(exp_steps),
        "--model", RIFE_MODEL_DIR
    ]
    
    print(f"[+] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[!] RIFE failed with return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    # Move generated frames to output directory
    if os.path.exists("output"):
        frames = sorted(glob.glob("output/*.png"))
        for i, frame_path in enumerate(frames):
            shutil.move(frame_path, f"rife_output/frame_{i:03d}.png")
        shutil.rmtree("output")
    
    print(f"[+] Generated {len(glob.glob('rife_output/*.png'))} interpolated frames")
    return True

# ==================== CONTROLNET STYLIZATION ====================

def load_stylization_models(control_type, device):
    """Load Stable Diffusion and ControlNet models."""
    print(f"[+] Loading models for {control_type} control...")
    
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL
        
        # Load ControlNet
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_PATHS[control_type], 
            torch_dtype=torch.float16
        )
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            VAE_PATH, 
            torch_dtype=torch.float16
        )
        
        # Load pipeline
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Enable memory efficient attention if available
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[+] xformers memory efficient attention enabled")
        except Exception:
            print("[!] xformers not available, using default attention")
        
        return pipe
        
    except Exception as e:
        print(f"[!] Error loading models: {e}")
        return None

def load_stylization_img2img_models(control_type, device):
    """Load Stable Diffusion ControlNet Img2Img pipeline for stronger background preservation."""
    print(f"[+] Loading Img2Img models for {control_type} control...")

    try:
        from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, AutoencoderKL

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_PATHS[control_type],
            torch_dtype=torch.float16
        )

        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            torch_dtype=torch.float16
        )

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)

        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[+] xformers memory efficient attention enabled")
        except Exception:
            print("[!] xformers not available, using default attention")

        return pipe

    except Exception as e:
        print(f"[!] Error loading img2img models: {e}")
        return None

def create_control_image(image, control_type):
    """Create control image (Canny, LineArt, or OpenPose) from input image."""
    try:
        if control_type == "canny":
            import cv2
            # Convert PIL to OpenCV format
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Apply Canny edge detection
            edges = cv2.Canny(img_cv, 100, 200)
            # Convert back to PIL
            control_img = Image.fromarray(edges).convert("RGB")
            
        elif control_type == "lineart":
            try:
                from controlnet_aux import LineartDetector
                detector = LineartDetector.from_pretrained("lllyasviel/Annotators")
                control_img = detector(image)
            except ImportError:
                print("[!] controlnet_aux not available, falling back to Canny")
                import cv2
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                edges = cv2.Canny(img_cv, 100, 200)
                control_img = Image.fromarray(edges).convert("RGB")
                
        elif control_type == "openpose":
            try:
                from controlnet_aux import OpenposeDetector
                detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                control_img = detector(image)
            except ImportError:
                print("[!] controlnet_aux not available, falling back to Canny")
                import cv2
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                edges = cv2.Canny(img_cv, 100, 200)
                control_img = Image.fromarray(edges).convert("RGB")
        
        return control_img
        
    except Exception as e:
        print(f"[!] Error creating control image: {e}")
        # Fallback to simple edge detection
        import cv2
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        edges = cv2.Canny(img_cv, 100, 200)
        return Image.fromarray(edges).convert("RGB")

def stylize_frames(middle_description, control_type="canny", num_inference_steps=20, 
                  guidance_scale=7.5, seed=42):
    """Stylize interpolated frames using ControlNet."""
    print(f"[+] Stylizing frames with description: '{middle_description}'")
    
    # Load models
    pipe = load_stylization_models(control_type, DEVICE)
    if pipe is None:
        return False
    
    # Set up generator
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Get list of interpolated frames
    frame_paths = sorted(glob.glob("rife_output/*.png"))
    if not frame_paths:
        print("[!] No interpolated frames found")
        return False
    
    print(f"[+] Stylizing {len(frame_paths)} frames...")
    
    # Process each frame
    for i, frame_path in enumerate(tqdm(frame_paths, desc="Stylizing")):
        # Load frame
        frame_img = Image.open(frame_path).convert("RGB")
        
        # Create control image
        control_img = create_control_image(frame_img, control_type)
        
        # Generate stylized image
        try:
            with torch.no_grad():
                result = pipe(
                    prompt=middle_description,
                    image=control_img,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=frame_img.height,
                    width=frame_img.width
                ).images[0]
            
            # Save stylized frame
            output_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
            result.save(output_path)
            
        except Exception as e:
            print(f"[!] Error stylizing frame {i}: {e}")
            # Save original frame as fallback
            output_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
            frame_img.save(output_path)
    
    print(f"[+] Stylized frames saved to {OUTPUT_DIR}/")
    return True

# ==================== SIMPLE LEFT-TO-RIGHT WALK ====================

def generate_linear_walk_frames(start_frame_path, num_frames=8, walk_ratio=0.4):
    """Generate a simple left-to-right motion by shifting the image each frame.

    The content is translated to the right by a constant step per frame.
    This ignores interpolation and stylization for a quick walking illusion.
    """
    print(f"[+] Generating linear walk frames: frames={num_frames}, walk_ratio={walk_ratio}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    img = Image.open(start_frame_path).convert("RGB")

    # Ensure dimensions divisible by 8 to be consistent with other steps
    width = ((img.width + 7) // 8) * 8
    height = ((img.height + 7) // 8) * 8
    if (width, height) != (img.width, img.height):
        img = resize_and_pad_image(img, (width, height))

    # Total horizontal shift in pixels across all frames
    total_shift_px = int(walk_ratio * width)
    step = total_shift_px / max(1, (num_frames - 1))

    for i in range(num_frames):
        dx = int(round(i * step))

        # Create black canvas and paste shifted image
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(img, (dx, 0))

        out_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
        canvas.save(out_path)

    print(f"[+] Linear walk frames saved to {OUTPUT_DIR}/")
    return True

def generate_pose_walk_frames(start_frame_path, num_frames=8, walk_ratio=0.4,
                              description="girl walking from left to right",
                              control_type="openpose",
                              num_inference_steps=20,
                              guidance_scale=7.5,
                              strength=0.35,
                              controlnet_scale=1.5,
                              seed=42):
    """Use ControlNet OpenPose with Img2Img to move the character while preserving background.

    - Creates an OpenPose control image from the start frame.
    - Translates the pose to the right a fixed amount each frame.
    - Runs SD ControlNet Img2Img with low strength to keep background consistent.
    """
    print(f"[+] Generating pose-walk frames: frames={num_frames}, walk_ratio={walk_ratio}, strength={strength}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    init_img = Image.open(start_frame_path).convert("RGB")

    # Normalize size to be divisible by 8
    width = ((init_img.width + 7) // 8) * 8
    height = ((init_img.height + 7) // 8) * 8
    if (width, height) != (init_img.width, init_img.height):
        init_img = resize_and_pad_image(init_img, (width, height))

    # Prepare OpenPose control image from the init image
    control_pose = create_control_image(init_img, control_type)
    control_pose = control_pose.resize((width, height), Image.NEAREST)

    # Load Img2Img pipeline
    pipe = load_stylization_img2img_models(control_type, DEVICE)
    if pipe is None:
        return False

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Compute shift per frame
    total_shift_px = int(walk_ratio * width)
    step = total_shift_px / max(1, (num_frames - 1))

    for i in range(num_frames):
        dx = int(round(i * step))

        # Shift the pose to the right on a black canvas (same size)
        shifted_pose = Image.new("RGB", (width, height), (0, 0, 0))
        shifted_pose.paste(control_pose, (dx, 0))

        try:
            with torch.no_grad():
                result = pipe(
                    prompt=description,
                    image=init_img,                 # init image to preserve background
                    control_image=shifted_pose,     # control pose shifted to the right
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                    guess_mode=False,
                    controlnet_conditioning_scale=controlnet_scale
                ).images[0]

            out_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
            result.save(out_path)
        except Exception as e:
            print(f"[!] Error in pose-walk frame {i}: {e}")
            # Fallback to saving the init image if generation fails
            out_path = os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png")
            init_img.save(out_path)

    print(f"[+] Pose-walk frames saved to {OUTPUT_DIR}/")
    return True

# ==================== MAIN PIPELINE ====================

def run_pipeline(middle_description, control_type="canny", num_inference_steps=20,
                guidance_scale=7.5, seed=42, num_frames=8):
    """Run the complete FrameForge pipeline."""
    return run_pipeline_with_paths(
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        middle_description=middle_description,
        control_type=control_type,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        num_frames=num_frames
    )

def run_pipeline_with_paths(start_frame, end_frame, middle_description, control_type="canny", 
                           num_inference_steps=20, guidance_scale=7.5, seed=42, num_frames=8):
    """Run the complete FrameForge pipeline with custom start/end frame paths."""
    global NUM_INTERPOLATED_FRAMES, START_FRAME, END_FRAME
    NUM_INTERPOLATED_FRAMES = num_frames
    START_FRAME = start_frame
    END_FRAME = end_frame
    
    print("[*] FrameForge Pipeline Starting...")
    print(f"[*] Description: {middle_description}")
    print(f"[*] Control Type: {control_type}")
    print(f"[*] Target Frames: {num_frames}")
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Start Frame: {start_frame}")
    print(f"[*] End Frame: {end_frame}")
    print("-" * 50)
    
    # Setup
    setup_directories()
    
    # Validate inputs
    if not validate_inputs():
        return False
    
    # Step 1: Prepare RIFE input
    if not prepare_rife_input():
        return False
    
    # Step 2: Run RIFE interpolation
    if not run_rife_interpolation():
        return False
    
    # Step 3: Stylize frames
    if not stylize_frames(middle_description, control_type, num_inference_steps, 
                         guidance_scale, seed):
        return False
    
    # Cleanup
    if os.path.exists("rife_input"):
        shutil.rmtree("rife_input")
    if os.path.exists("rife_output"):
        shutil.rmtree("rife_output")
    
    print("-" * 50)
    print("[+] Pipeline completed successfully!")
    print(f"[+] Output frames saved in: {os.path.abspath(OUTPUT_DIR)}")
    
    return True

# ==================== COMMAND LINE INTERFACE ====================

def main():
    parser = argparse.ArgumentParser(description="FrameForge: AI-Powered Frame Interpolation and Stylization")
    
    parser.add_argument("--description", "-d", required=True,
                       help="Description of what happens between start and end frames")
    parser.add_argument("--control-type", "-c", choices=["canny", "lineart", "openpose"],
                       default=DEFAULT_CONTROL_TYPE, help="ControlNet type to use")
    parser.add_argument("--num-frames", "-n", type=int, default=8,
                       help="Number of frames to generate (default: 8)")
    parser.add_argument("--inference-steps", "-i", type=int, default=DEFAULT_NUM_INFERENCE_STEPS,
                       help="Number of inference steps (default: 20)")
    parser.add_argument("--guidance-scale", "-g", type=float, default=DEFAULT_GUIDANCE_SCALE,
                       help="Guidance scale (default: 7.5)")
    parser.add_argument("--seed", "-s", type=int, default=DEFAULT_SEED,
                       help="Random seed (default: 42)")
    parser.add_argument("--start-frame", default=START_FRAME,
                       help="Path to start frame (default: data/start_frame.png)")
    parser.add_argument("--end-frame", default=END_FRAME,
                       help="Path to end frame (default: data/end_frame.png)")
    # Simple linear walk mode (bypass interpolation and stylization)
    parser.add_argument("--linear-walk", action="store_true",
                       help="Generate frames by shifting the start image left-to-right")
    parser.add_argument("--walk-ratio", type=float, default=0.4,
                       help="Fraction of width to traverse across all frames (default: 0.4)")
    # Pose-walk mode using ControlNet OpenPose with img2img
    parser.add_argument("--pose-walk", action="store_true",
                       help="Use ControlNet OpenPose img2img to move the character while preserving background")
    parser.add_argument("--pose-strength", type=float, default=0.35,
                       help="Img2Img strength for background preservation (lower preserves more)")
    parser.add_argument("--pose-scale", type=float, default=1.5,
                       help="ControlNet conditioning scale (higher forces pose adherence)")
    
    args = parser.parse_args()
    
    # If linear walk mode, bypass interpolation and stylization
    if args.pose_walk:
        # Force openpose for pose-walk regardless of control-type
        success = generate_pose_walk_frames(
            start_frame_path=args.start_frame,
            num_frames=args.num_frames,
            walk_ratio=args.walk_ratio,
            description=args.description,
            control_type="openpose",
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.pose_strength,
            controlnet_scale=args.pose_scale,
            seed=args.seed,
        )
    elif args.linear_walk:
        success = generate_linear_walk_frames(
            start_frame_path=args.start_frame,
            num_frames=args.num_frames,
            walk_ratio=args.walk_ratio,
        )
    else:
        # Run pipeline with custom paths
        success = run_pipeline_with_paths(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            middle_description=args.description,
            control_type=args.control_type,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            num_frames=args.num_frames
        )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
