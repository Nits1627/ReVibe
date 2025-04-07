# app.py

import gradio as gr
from PIL import Image
import traceback
import sys
import torch
import os
import numpy as np
from typing import Optional, Tuple, Dict, List, Any

# Import our modules
from classification import load_vit_model, classify_style, IDX_TO_CLASS
from prompting import generate_prompt
from generation import load_sd_pipeline, generate_sd_image, load_inpaint_pipeline, inpaint_image

# -------------------------------
# Setup: Device and Model Loading
# -------------------------------

# To avoid potential issues with MPS, you can use "cpu". If your system supports MPS reliably, you can change this.
device = "cpu"  # or "mps" if you wish to try MPS

# Flag to control whether to use fine-tuned models
USE_FINETUNED_VIT = True  # Use fine-tuned ViT model
USE_FINETUNED_SD = False  # Skip fine-tuned SD model (takes too long to train)

# Load the ViT classifier and processor with fine-tuning option
vit_model, processor = load_vit_model(model_name="google/vit-base-patch16-224", use_finetuned=USE_FINETUNED_VIT)

# Load the Stable Diffusion pipeline with fine-tuning option
sd_pipe = load_sd_pipeline(device=device, use_finetuned=USE_FINETUNED_SD)

# Load the inpainting pipeline with fine-tuning option
inpaint_pipe = load_inpaint_pipeline(device=device, use_finetuned=USE_FINETUNED_SD)

# -------------------------------
# Style Examples Resources
# -------------------------------

# Define paths to style example images
# These should be created and placed in a 'style_examples' directory
STYLE_EXAMPLES_DIR = "style_examples"
os.makedirs(STYLE_EXAMPLES_DIR, exist_ok=True)

def get_style_example_path(style_name: str) -> str:
    """Get the file path for a style example image."""
    return os.path.join(STYLE_EXAMPLES_DIR, f"{style_name.lower()}_example.jpg")

# -------------------------------
# Define Common Room Elements for Inpainting
# -------------------------------

ROOM_ELEMENTS = [
    "Sofa",
    "Coffee table",
    "Bookshelf",
    "Dining table",
    "Chair",
    "Bed",
    "Desk",
    "Lamp",
    "TV stand",
    "Cabinet",
    "Wall art",
    "Carpet/Rug",
    "Plants",
    "Curtains/Blinds"
]

ELEMENT_COLORS = [
    "White",
    "Black",
    "Brown",
    "Gray",
    "Blue",
    "Green",
    "Red",
    "Yellow",
    "Purple",
    "Pink",
    "Orange",
    "Teal",
    "Navy",
    "Beige",
    "Wood finish"
]

ELEMENT_ACTIONS = [
    "Add",
    "Replace",
    "Remove",
    "Change color"
]

# -------------------------------
# Define the Full Processing Functions
# -------------------------------

def update_prompt(current_style: str, target_style: str, custom_prompt: Optional[str] = None) -> str:
    """Generate or use a custom prompt."""
    if custom_prompt and custom_prompt.strip():
        return custom_prompt.strip()
    return generate_prompt(current_style, target_style)

def process_image(
    uploaded_image: Image.Image, 
    target_style: str,
    custom_prompt: str = "",
    use_custom_prompt: bool = False
) -> Tuple[str, str, Image.Image, gr.Gallery]:
    """
    Full pipeline:
      1. Classify the uploaded image to detect the current style.
      2. Generate a prompt based on the detected style and the target style.
      3. Generate a new interior design image using Stable Diffusion.
      
    Returns:
      - Predicted current style
      - Generated/custom prompt
      - The generated image
      - A gallery with original and generated images for comparison
    """
    try:
        # Detect the current style
        current_style = classify_style(uploaded_image, vit_model, processor)
        
        # Generate or use custom prompt
        prompt = custom_prompt if use_custom_prompt and custom_prompt.strip() else generate_prompt(current_style, target_style)
        
        # Generate the new image
        generated_image = generate_sd_image(sd_pipe, prompt)
        
        # Return both images for comparison
        comparison_gallery = [
            (uploaded_image, "Original Image"),
            (generated_image, "Generated Image")
        ]
        
        return current_style, prompt, generated_image, comparison_gallery
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return f"Error: {str(e)}", "", None, None

def generate_inpainting_prompt(action, element, color=None):
    """
    Generate a prompt for inpainting based on the action, element, and optional color.
    """
    if action == "Add":
        return f"Add a {color + ' ' if color else ''}modern {element} to this room"
    elif action == "Replace":
        return f"Replace the existing {element} with a {color + ' ' if color else ''}modern one"
    elif action == "Remove":
        return f"Remove the {element} from this room, leaving an appropriate space"
    elif action == "Change color":
        return f"Change the color of the {element} to {color}" if color else f"Change the {element} to a different color"
    return ""

def process_inpainting(
    image: Image.Image,
    mask: Image.Image,
    action: str,
    element: str,
    color: Optional[str] = None,
    custom_prompt: str = ""
) -> Tuple[Image.Image, gr.Gallery]:
    """
    Process inpainting operation:
      1. Generate a prompt based on selected action, element, and color
      2. Apply inpainting to the masked region
      
    Returns:
      - The inpainted image
      - A gallery with original and inpainted images for comparison
    """
    try:
        # Validate inputs
        if image is None:
            raise ValueError("No image provided. Please upload an image.")
        if mask is None:
            raise ValueError("No mask provided. Please draw on the mask to indicate the area to modify.")
            
        # Ensure mask is in the right format
        # Convert to RGB as required by the model
        mask_image = mask.convert('RGB')
        
        # Use custom prompt if provided, otherwise generate one based on inputs
        prompt = custom_prompt if custom_prompt.strip() else generate_inpainting_prompt(action, element, color)
        
        # Perform inpainting
        inpainted_image = inpaint_image(inpaint_pipe, prompt, image, mask_image)
        
        # Return both images for comparison
        comparison_gallery = [
            (image, "Original Image"),
            (inpainted_image, "Inpainted Image")
        ]
        
        return inpainted_image, comparison_gallery
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return None, None

def on_style_select(style: str):
    """Update the style example image when a style is selected."""
    example_path = get_style_example_path(style)
    if os.path.exists(example_path):
        return example_path
    return None

def on_detect_style(image: Image.Image) -> str:
    """Detect the style of the uploaded image."""
    if image is None:
        return ""
    try:
        style = classify_style(image, vit_model, processor)
        return style
    except:
        return "Unable to detect style"

def on_action_select(action):
    """Update UI components based on the selected action"""
    # If action is "Change color", show color dropdown
    color_visible = (action == "Change color")
    return gr.update(visible=color_visible)

# -------------------------------
# Build the Gradio Interface
# -------------------------------

# List of target styles must correspond to the available classes.
target_styles_list = list(IDX_TO_CLASS.values())

with gr.Blocks(title="Interior Design Generator & Editor") as iface:
    gr.Markdown("# Interior Design Generator & Editor")
    gr.Markdown("Upload an image of your room to either transform its style or make specific modifications to elements.")
    
    # Add information about fine-tuned models
    if USE_FINETUNED_VIT:
        gr.Markdown("### Using Fine-tuned Models")
        gr.Markdown("This application uses models that have been fine-tuned on a dataset of interior design images across 8 different styles.")
    
    with gr.Tabs() as tabs:
        # Tab 1: Style Transformation
        with gr.TabItem("Style Transformation"):
            gr.Markdown("Transform the entire style of your room.")
            
            with gr.Row():
                # Left column - Input
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Room Image")
                    detect_button = gr.Button("Detect Current Style")
                    current_style_text = gr.Textbox(label="Detected Style", interactive=False)
                    
                    # Style selection with examples
                    with gr.Row():
                        target_style_dropdown = gr.Dropdown(
                            choices=target_styles_list, 
                            label="Select Target Style",
                            value=target_styles_list[0] if target_styles_list else None
                        )
                        style_example_image = gr.Image(label="Style Example", interactive=False)
                    
                    # Prompt editing
                    with gr.Accordion("Advanced Options", open=False):
                        generated_prompt = gr.Textbox(label="Generated Prompt", interactive=False)
                        use_custom_prompt = gr.Checkbox(label="Edit Prompt", value=False)
                        custom_prompt = gr.Textbox(
                            label="Custom Prompt", 
                            placeholder="Edit the prompt here if you want to customize...",
                            interactive=True,
                            visible=True
                        )
                    
                    generate_button = gr.Button("Generate Design", variant="primary")
                
                # Right column - Output
                with gr.Column(scale=1):
                    with gr.Tab("Side-by-Side Comparison"):
                        comparison_gallery = gr.Gallery(
                            label="Before & After", 
                            show_label=True,
                            elem_id="comparison_gallery",
                            columns=2,
                            height=400
                        )
                    
                    with gr.Tab("Generated Image"):
                        output_image = gr.Image(label="Generated Interior Design")
        
        # Tab 2: Element Inpainting
        with gr.TabItem("Element Editing"):
            gr.Markdown("Make specific changes to elements in your room.")
            
            with gr.Row():
                # Left column - Input for inpainting
                with gr.Column(scale=1):
                    # Use separate image components for inpainting
                    inpaint_input_image = gr.Image(type="pil", label="Upload Room Image")
                    inpaint_mask = gr.Image(type="pil", label="Upload or Create Mask (white areas will be changed)")
                    gr.Markdown("Upload your room image, then create or upload a mask where white areas indicate regions to modify.")
                    
                    with gr.Row():
                        action_dropdown = gr.Dropdown(
                            choices=ELEMENT_ACTIONS,
                            label="Select Action",
                            value=ELEMENT_ACTIONS[0]
                        )
                        element_dropdown = gr.Dropdown(
                            choices=ROOM_ELEMENTS,
                            label="Select Element",
                            value=ROOM_ELEMENTS[0]
                        )
                    
                    color_dropdown = gr.Dropdown(
                        choices=ELEMENT_COLORS,
                        label="Select Color",
                        value=ELEMENT_COLORS[0],
                        visible=False  # Initially hidden, shown only for "Change color" action
                    )
                    
                    # Custom prompt for more control
                    with gr.Accordion("Advanced Options", open=False):
                        inpaint_generated_prompt = gr.Textbox(
                            label="Generated Prompt", 
                            interactive=False,
                            value=""
                        )
                        use_inpaint_custom_prompt = gr.Checkbox(label="Edit Prompt", value=False)
                        inpaint_custom_prompt = gr.Textbox(
                            label="Custom Inpainting Prompt", 
                            placeholder="Edit the prompt here for precise control...",
                            interactive=True,
                            visible=True
                        )
                    
                    inpaint_button = gr.Button("Apply Changes", variant="primary")
                
                # Right column - Output for inpainting
                with gr.Column(scale=1):
                    with gr.Tab("Side-by-Side Comparison"):
                        inpaint_comparison_gallery = gr.Gallery(
                            label="Before & After", 
                            show_label=True,
                            elem_id="inpaint_comparison_gallery",
                            columns=2,
                            height=400
                        )
                    
                    with gr.Tab("Edited Image"):
                        inpaint_output_image = gr.Image(label="Edited Room")
    
    # Set up event handlers for Style Transformation
    detect_button.click(
        fn=on_detect_style,
        inputs=[input_image],
        outputs=[current_style_text]
    )
    
    target_style_dropdown.change(
        fn=on_style_select,
        inputs=[target_style_dropdown],
        outputs=[style_example_image]
    )
    
    input_image.change(
        fn=on_detect_style,
        inputs=[input_image],
        outputs=[current_style_text]
    )
    
    # When detect style button is clicked, also generate the prompt
    detect_button.click(
        fn=lambda img, style: generate_prompt(on_detect_style(img), style) if img is not None and style else "",
        inputs=[input_image, target_style_dropdown],
        outputs=[generated_prompt]
    )
    
    # When target style changes, update the prompt
    target_style_dropdown.change(
        fn=lambda img, style: generate_prompt(on_detect_style(img), style) if img is not None and style else "",
        inputs=[input_image, target_style_dropdown],
        outputs=[generated_prompt]
    )
    
    # Handle the main generation process
    generate_button.click(
        fn=process_image,
        inputs=[
            input_image,
            target_style_dropdown,
            custom_prompt,
            use_custom_prompt
        ],
        outputs=[
            current_style_text,
            generated_prompt,
            output_image,
            comparison_gallery
        ]
    )
    
    # Set up event handlers for Element Inpainting
    # Update visibility of color dropdown based on action
    action_dropdown.change(
        fn=on_action_select,
        inputs=[action_dropdown],
        outputs=[color_dropdown]
    )
    
    # Update the inpainting prompt when options change
    def update_inpaint_prompt(action, element, color):
        return generate_inpainting_prompt(action, element, color)
    
    # When inpainting options change, update the prompt
    for input_component in [action_dropdown, element_dropdown, color_dropdown]:
        input_component.change(
            fn=update_inpaint_prompt,
            inputs=[action_dropdown, element_dropdown, color_dropdown],
            outputs=[inpaint_generated_prompt]
        )
    
    # Handle the inpainting process
    inpaint_button.click(
        fn=lambda img, mask, action, element, color, custom_prompt, use_custom: 
            process_inpainting(
                img, mask, action, element, color, 
                custom_prompt if use_custom else ""
            ),
        inputs=[
            inpaint_input_image,
            inpaint_mask,
            action_dropdown,
            element_dropdown,
            color_dropdown,
            inpaint_custom_prompt,
            use_inpaint_custom_prompt
        ],
        outputs=[
            inpaint_output_image,
            inpaint_comparison_gallery
        ]
    )

if __name__ == "__main__":
    # share=True creates a public shareable link if localhost is not accessible.
    iface.launch(share=True)


