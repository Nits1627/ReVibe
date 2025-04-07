# generation.py

import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, StableDiffusionInpaintPipeline
import os
from PIL import Image
from datetime import datetime

# Flag to control whether to use fine-tuned models
USE_FINETUNED_SD = False  # Set to False to skip using fine-tuned SD model

def load_sd_pipeline(model_name="runwayml/stable-diffusion-v1-5", device="cpu", use_finetuned=USE_FINETUNED_SD):
    """
    Load the Stable Diffusion pipeline.
    If use_finetuned is True, load our fine-tuned model instead.
    """
    try:
        # Check if fine-tuned model exists
        finetuned_path = "sd_finetuned_model"
        
        if use_finetuned and os.path.exists(finetuned_path) and os.path.isdir(finetuned_path) and len(os.listdir(finetuned_path)) > 0:
            print(f"Loading fine-tuned Stable Diffusion model from {finetuned_path}")
            pipe = StableDiffusionPipeline.from_pretrained(
                finetuned_path,
                torch_dtype=torch.float32,
                safety_checker=None
            )
        else:
            if use_finetuned:
                print(f"Fine-tuned model not found or empty directory. Loading pre-trained model instead.")
            else:
                print(f"Loading pre-trained Stable Diffusion model from {model_name}")
                
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None
            )
        
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        
        # Enable attention slicing for memory efficiency
        pipe.enable_attention_slicing()
        
        # Move to the specified device
        pipe.to(device)
        return pipe
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        raise

def load_inpaint_pipeline(model_name="runwayml/stable-diffusion-inpainting", device="cpu", use_finetuned=USE_FINETUNED_SD):
    """
    Load the Stable Diffusion inpainting pipeline.
    If use_finetuned is True, attempt to load our fine-tuned model for inpainting.
    """
    try:
        # For inpainting, we typically use a specialized pre-trained model
        # Fine-tuned model for inpainting would need special training or adaptation
        # Currently just check if a specific inpainting fine-tuned model exists
        finetuned_inpaint_path = "sd_finetuned_inpaint_model"
        
        if use_finetuned and os.path.exists(finetuned_inpaint_path) and os.path.isdir(finetuned_inpaint_path) and len(os.listdir(finetuned_inpaint_path)) > 0:
            print(f"Loading fine-tuned inpainting model from {finetuned_inpaint_path}")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                finetuned_inpaint_path,
                torch_dtype=torch.float32,
                safety_checker=None
            )
        else:
            print(f"Loading pre-trained inpainting model from {model_name}")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None
            )
        
        # Enable attention slicing for memory efficiency
        pipe.enable_attention_slicing()
        
        # Move to the specified device
        pipe.to(device)
        return pipe
    except Exception as e:
        print(f"Error loading inpainting pipeline: {e}")
        raise

def generate_sd_image(pipe, prompt: str, num_inference_steps=25, guidance_scale=7.5):
    """
    Generate an image using Stable Diffusion given a prompt.
    """
    try:
        with torch.no_grad():
            image = pipe(
                prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        raise

def inpaint_image(pipe, prompt: str, image: Image.Image, mask_image: Image.Image, 
                 num_inference_steps=25, guidance_scale=7.5):
    """
    Inpaint an image using Stable Diffusion Inpainting given a prompt, 
    original image, and mask image.
    
    Parameters:
    - pipe: The inpainting pipeline
    - prompt: Text prompt describing what to add/change in the masked area
    - image: Original image to modify
    - mask_image: Black and white mask where white indicates areas to inpaint
    - num_inference_steps: Number of denoising steps
    - guidance_scale: Higher values guide the model more strongly towards the prompt
    
    Returns:
    - Inpainted image
    """
    try:
        # Ensure images are in RGB format
        image = image.convert("RGB")
        mask_image = mask_image.convert("RGB")
        
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            inpainted_image = output.images[0]
            
        return inpainted_image
    except Exception as e:
        print(f"Error inpainting image: {e}")
        raise

def save_image(image, output_dir="outputs", prefix="generated"):
    """
    Save the generated image with a timestamp.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save the image
        image.save(filepath)
        print(f"Image saved to {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        raise

# Test the image generation when the script is run directly
if __name__ == "__main__":
    try:
        # Determine if CUDA is available and set device accordingly
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load the pipeline
        print("Loading Stable Diffusion pipeline...")
        pipe = load_sd_pipeline(device=device, use_finetuned=True)
        print("Pipeline loaded successfully!")
        
        # Generate a test image
        test_prompt = "A cozy modern living room with large windows, minimal furniture, and plants"
        print(f"Generating image with prompt: \"{test_prompt}\"")
        
        image = generate_sd_image(pipe, test_prompt)
        
        # Save the generated image
        save_path = save_image(image)
        
        print("Image generation completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")