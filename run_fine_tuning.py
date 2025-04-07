#!/usr/bin/env python3
# run_fine_tuning.py

import os
import argparse
import subprocess
import sys
from pathlib import Path
import time
import shutil

def check_dependencies():
    """Check if all the necessary dependencies are installed."""
    try:
        import tensorflow as tf
        import torch
        import transformers
        import diffusers
        import accelerate
        from PIL import Image
        from tqdm import tqdm
        print("All dependencies found!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required packages:")
        print("pip install tensorflow torch transformers diffusers accelerate Pillow tqdm")
        return False

def check_dataset():
    """Check if the dataset exists and has the expected structure."""
    dataset_path = Path("sem6_atml_ds")
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} not found.")
        return False
        
    # Expected style directories
    expected_styles = [
        "transitional", 
        "modern", 
        "minimalist", 
        "industrial", 
        "coastal", 
        "scandinavian", 
        "bohemian", 
        "mid-century"
    ]
    
    missing_styles = []
    for style in expected_styles:
        style_path = dataset_path / style
        if not style_path.exists() or not style_path.is_dir():
            missing_styles.append(style)
    
    if missing_styles:
        print(f"Error: Missing style directories: {', '.join(missing_styles)}")
        return False
        
    # Check that each directory has some images
    empty_styles = []
    for style in expected_styles:
        style_path = dataset_path / style
        if style_path.exists():
            images = list(style_path.glob("*.jpg")) + list(style_path.glob("*.jpeg")) + list(style_path.glob("*.png"))
            if not images:
                empty_styles.append(style)
    
    if empty_styles:
        print(f"Warning: These style directories have no images: {', '.join(empty_styles)}")
        
    print("Dataset structure looks good!")
    return True

def run_vit_fine_tuning(args):
    """Run the ViT model fine-tuning process."""
    print("\n" + "="*50)
    print("Starting ViT fine-tuning for interior design style classification")
    print("="*50)
    
    # Clean up existing model directory to avoid issues with previous failed runs
    vit_model_dir = "vit_finetuned_model"
    if os.path.exists(vit_model_dir):
        print(f"Removing existing directory: {vit_model_dir}")
        shutil.rmtree(vit_model_dir)
    os.makedirs(vit_model_dir, exist_ok=True)
    
    command = [
        sys.executable, 
        "fine_tune_vit.py"
    ]
    
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
    start_time = time.time()
    try:
        process = subprocess.run(command, check=True)
        end_time = time.time()
        print(f"ViT fine-tuning completed in {(end_time - start_time)/60:.2f} minutes!")
        return process.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error during ViT fine-tuning: {e}")
        print("Check the error message above for details.")
        return False

def run_sd_fine_tuning(args):
    """Run the Stable Diffusion fine-tuning process."""
    print("\n" + "="*50)
    print("Starting Stable Diffusion fine-tuning for interior design generation")
    print("="*50)
    
    # Force CPU for training to avoid MPS/CUDA issues
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ACCELERATE_USE_MPS"] = "false"
    os.environ["ACCELERATE_USE_CUDA"] = "false"
    
    command = [
        sys.executable, 
        "fine_tune_sd.py",
        "--num_train_epochs", str(args.epochs),
        "--train_batch_size", "1",  # Keep batch size small due to memory constraints
    ]
    
    # GPU flags are ignored since we're forcing CPU
    # This is to ensure compatibility across platforms
    
    start_time = time.time()
    process = subprocess.run(command, check=True)
    end_time = time.time()
    
    print(f"Stable Diffusion fine-tuning completed in {(end_time - start_time)/60:.2f} minutes!")
    return process.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuning workflow for interior design models.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for SD fine-tuning")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and dataset checks")
    parser.add_argument("--vit-only", action="store_true", help="Only fine-tune the ViT model")
    parser.add_argument("--sd-only", action="store_true", help="Only fine-tune the Stable Diffusion model")
    parser.add_argument("--run-sd", action="store_true", help="Run Stable Diffusion fine-tuning (disabled by default)")
    args = parser.parse_args()
    
    print("="*50)
    print("Interior Design Style Fine-tuning Workflow")
    print("="*50)
    
    try:
        # Check dependencies and dataset
        if not args.skip_checks:
            if not check_dependencies():
                return 1
            if not check_dataset():
                return 1
                
        # Create output directories
        os.makedirs("sd_finetuned_model", exist_ok=True)
        
        # Run fine-tuning processes
        success = True
        
        if not args.sd_only:
            success = run_vit_fine_tuning(args)
            if not success:
                print("Error: ViT fine-tuning failed!")
                return 1
        
        # By default, skip Stable Diffusion fine-tuning unless explicitly requested
        if (not args.vit_only and success) and (args.run_sd or args.sd_only):
            print("\nRunning Stable Diffusion fine-tuning (this may take several hours)...")
            success = run_sd_fine_tuning(args)
            if not success:
                print("Error: Stable Diffusion fine-tuning failed!")
                return 1
        else:
            if not args.sd_only:
                print("\nSkipping Stable Diffusion fine-tuning. Use --run-sd flag to enable it.")
                print("Note: Stable Diffusion fine-tuning can take 8+ hours on CPU.")
        
        if success:
            print("\n" + "="*50)
            print("Fine-tuning workflow completed successfully!")
            print("You can now use the fine-tuned models in the application.")
            print("="*50)
            return 0
        return 1
        
    except Exception as e:
        print(f"\nUnexpected error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 