#!/usr/bin/env python3
# fine_tune_sd.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import random
from tqdm import tqdm
import argparse
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
import shutil
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# Check available device and set accordingly
# Force CPU usage to avoid MPS-related issues on Mac
device = "cpu"
print(f"Forcing device: {device}")

# Our interior design style classes
STYLES = [
    "transitional",
    "modern",
    "minimalist",
    "industrial",
    "coastal", 
    "scandinavian",
    "bohemian",
    "mid-century"
]

# Dataset path
DATASET_PATH = Path("sem6_atml_ds")
OUTPUT_DIR = Path("sd_finetuned_model")

class InteriorDesignDataset(Dataset):
    """Dataset for interior design images."""
    
    def __init__(self, tokenizer, size=512, center_crop=True):
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        
        self.image_paths = []
        self.captions = []
        
        # Collect all images and generate captions
        for style in STYLES:
            style_path = DATASET_PATH / style
            if not style_path.exists() or not style_path.is_dir():
                continue
                
            print(f"Processing {style} images...")
            style_images = list(style_path.glob("*.jpg")) + list(style_path.glob("*.jpeg")) + list(style_path.glob("*.png"))
            
            for img_path in tqdm(style_images):
                # Create a caption for the image
                caption = self._generate_caption(style, img_path.stem)
                
                self.image_paths.append(img_path)
                self.captions.append(caption)
        
        print(f"Dataset loaded: {len(self.image_paths)} images")
    
    def _generate_caption(self, style, filename):
        """Generate a caption for an interior design image."""
        # Base captions for different styles
        style_captions = {
            "transitional": [
                f"A {style} living room with classic and contemporary elements, neutral colors.",
                f"A {style} interior with clean lines, traditional details, and neutral palette.",
                f"A comfortable {style} living space with timeless furniture and subtle patterns."
            ],
            "modern": [
                f"A {style} room with clean lines, minimal accessories, and bold colors.",
                f"A sleek {style} interior with open space, minimal clutter, and geometric forms.",
                f"A contemporary {style} living space with innovative furniture and technology integration."
            ],
            "minimalist": [
                f"A {style} interior with essential furniture, clean lines, and monochromatic palette.",
                f"A clean {style} living space with limited color palette and functional design.",
                f"A spacious {style} room with open layout, few furniture pieces, and hidden storage."
            ],
            "industrial": [
                f"An {style} interior with exposed brick, metal finishes, and raw elements.",
                f"A {style} living space with factory-inspired features, pipes, and concrete surfaces.",
                f"An urban {style} room with open ductwork, weathered wood, and utilitarian objects."
            ],
            "coastal": [
                f"A {style} interior with light blue and white color scheme, natural light.",
                f"A {style} living space with beach-inspired elements, light woods, and airy fabrics.",
                f"A bright {style} room with nautical accents, white furniture, and ocean-inspired palette."
            ],
            "scandinavian": [
                f"A {style} interior with light wood, neutral colors, and simple functional furniture.",
                f"A bright {style} living space with minimal decoration, white walls, and wooden elements.",
                f"A cozy {style} room with clean lines, natural materials, and practical design."
            ],
            "bohemian": [
                f"A {style} interior with eclectic mix of patterns, textures, and global influences.",
                f"A colorful {style} living space with layered textiles, plants, and diverse accessories.",
                f"A free-spirited {style} room with vintage furniture, rich colors, and ethnic elements."
            ],
            "mid-century": [
                f"A {style} interior with iconic furniture, organic shapes, and functional design.",
                f"A retro {style} living space with clean lines, graphic patterns, and warm woods.",
                f"A vintage-inspired {style} room with iconic chairs, wooden elements, and pops of color."
            ]
        }
        
        # Get captions for the style or use generic ones
        captions = style_captions.get(style, [f"A beautiful {style} interior design", f"A {style} decorated room"])
        
        # Add some randomness - pick one of the captions
        return random.choice(captions)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Resize and center crop
            if self.center_crop:
                width, height = image.size
                min_dim = min(width, height)
                image = image.crop(
                    ((width - min_dim) // 2,
                     (height - min_dim) // 2,
                     (width + min_dim) // 2,
                     (height + min_dim) // 2)
                )
            
            # Resize to the desired size
            image = image.resize((self.size, self.size), Image.LANCZOS)
            
            # Convert to numpy array and normalize
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # [C, H, W]
            
            # Normalize to [-1, 1]
            image_tensor = 2.0 * image_tensor - 1.0
            
            # Tokenize caption
            caption_tokens = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]
            
            return {
                "pixel_values": image_tensor,
                "input_ids": caption_tokens,
                "caption": caption
            }
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # If this is the only item or we've tried all items, return a default
            if len(self.image_paths) <= 1 or idx == (len(self.image_paths) - 1):
                print("Cannot find a valid image, creating a dummy tensor")
                # Return dummy data
                return {
                    "pixel_values": torch.zeros((3, self.size, self.size)),
                    "input_ids": torch.zeros(self.tokenizer.model_max_length, dtype=torch.long),
                    "caption": "Error processing image"
                }
            # Try the next item
            return self.__getitem__((idx + 1) % len(self.image_paths))

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Disable accelerator's automatic device placement
    os.environ["ACCELERATE_USE_MPS"] = "false"
    os.environ["ACCELERATE_USE_CUDA"] = "false"
    
    # Force CPU for all PyTorch operations
    torch.device('cpu')
    
    # Setup accelerator with CPU device
    project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="no",  # No mixed precision on CPU
        project_config=project_config,
        cpu=True  # Force CPU
    )
    
    print(f"Accelerator device: {accelerator.device}")
    
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    
    # Move text encoder to CPU explicitly
    text_encoder = text_encoder.to("cpu")
    
    # Create dataset and dataloader
    dataset = InteriorDesignDataset(tokenizer, size=args.resolution)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers
    )
    
    # Load the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
        torch_dtype=torch.float32,  # Use float32 to avoid precision issues
    )
    
    # Set the scheduler
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    
    # Force all model components to CPU
    pipeline.to("cpu")
    
    # Use attention slicing to reduce memory usage
    pipeline.enable_attention_slicing()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        pipeline.unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Setup LR scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs,
    )
    
    # Prepare for training
    pipeline.unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipeline.unet, optimizer, dataloader, lr_scheduler
    )
    
    # Text encoder doesn't require gradient
    text_encoder.requires_grad_(False)
    
    # Set training mode for unet
    pipeline.unet.train()
    
    # Training loop
    for epoch in range(args.num_train_epochs):
        pipeline.unet.train()
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(dataloader):
            # Make sure inputs are on the correct device
            pixel_values = batch["pixel_values"].to("cpu")
            input_ids = batch["input_ids"].to("cpu")
            
            # Convert images to latent space
            latents = pipeline.vae.encode(pixel_values).latent_dist.sample() * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(input_ids)[0]
            
            # Predict the noise residual
            noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update parameters
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            
        # End of epoch
        accelerator.wait_for_everyone()
        
        # Save the checkpoint
        if accelerator.is_main_process:
            # Unwrap vae and unet
            unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
            
            # Create a new pipeline with the fine-tuned models
            save_pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unwrapped_unet,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                safety_checker=None,
            )
            
            # Save the pipeline
            save_pipeline.save_pretrained(args.output_dir)
            
            # Generate test images
            try:
                save_pipeline.unet.eval()
                with torch.no_grad():
                    for style in STYLES:
                        test_prompt = f"A beautiful {style} interior design living room"
                        image = save_pipeline(test_prompt).images[0]
                        os.makedirs(os.path.join(args.output_dir, "test_samples"), exist_ok=True)
                        image.save(os.path.join(args.output_dir, "test_samples", f"{style}_epoch_{epoch}.png"))
            except Exception as e:
                print(f"Error generating test images: {e}")
    
    # End of training
    accelerator.wait_for_everyone()
    
    # Final save
    if accelerator.is_main_process:
        # Unwrap vae and unet
        unwrapped_unet = accelerator.unwrap_model(pipeline.unet)
        
        # Create a new pipeline with the fine-tuned models
        save_pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unwrapped_unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
        )
        
        # Save the pipeline
        save_pipeline.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion on interior design images.")
    
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_dir", type=str, default="sd_finetuned_model")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    main(args) 