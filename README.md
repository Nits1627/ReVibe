# Interior Design Style Classifier and Generator

This project provides tools to classify interior design styles and generate new room designs based on specific styles.

## What's New: Fine-tuning Support

We've added support for fine-tuning both the ViT classifier and Stable Diffusion model on our dataset of interior design styles.

### Dataset Structure

The dataset is organized in the following structure:
```
sem6_atml_ds/
  ├── transitional/
  ├── modern/
  ├── minimalist/
  ├── industrial/
  ├── coastal/
  ├── scandinavian/
  ├── bohemian/
  └── mid-century/
```

Each style directory contains images that represent that particular interior design style.

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Fine-tuning the Models

You can fine-tune both models using the provided script:

```bash
python run_fine_tuning.py
```

### Command-line Options

The fine-tuning script supports the following options:

- `--gpu`: Use GPU for training (recommended for Stable Diffusion fine-tuning)
- `--epochs`: Number of epochs for Stable Diffusion fine-tuning (default: 1)
- `--skip-checks`: Skip dependency and dataset checks
- `--vit-only`: Only fine-tune the ViT model
- `--sd-only`: Only fine-tune the Stable Diffusion model

Examples:

```bash
# Fine-tune both models using GPU
python run_fine_tuning.py --gpu

# Fine-tune only the ViT model
python run_fine_tuning.py --vit-only

# Fine-tune Stable Diffusion for 3 epochs
python run_fine_tuning.py --sd-only --epochs 3
```

## Fine-tuning Details

### Vision Transformer (ViT)

The ViT model is fine-tuned for the classification task. The script:
1. Loads the pre-trained ViT model
2. Processes the images from each style directory
3. Splits the data into training, validation, and test sets
4. Fine-tunes the model on the training data
5. Evaluates the model on the test set
6. Saves the fine-tuned model to the `vit_finetuned_model` directory

### Stable Diffusion

The Stable Diffusion model is fine-tuned for generating interior design images. The script:
1. Loads the pre-trained Stable Diffusion model
2. Creates a dataset with images and corresponding style captions
3. Fine-tunes the model on the dataset
4. Generates example images for each style
5. Saves the fine-tuned model to the `sd_finetuned_model` directory

## Using the Fine-tuned Models

After fine-tuning, the models will be used automatically by the application. The `USE_FINETUNED_MODELS` flag in `app.py` controls whether to use the fine-tuned models or the pre-trained ones.

To use the application:

```bash
python app.py
```

## Features

- Classify interior design styles from uploaded images
- Generate new interior designs based on a target style
- Inpaint/modify specific elements in a room
- Fine-tune models on your own dataset of interior design styles

## Supported Styles

- Transitional
- Modern
- Minimalist
- Industrial
- Coastal
- Scandinavian
- Bohemian
- Mid-Century 




