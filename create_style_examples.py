# create_style_examples.py
import os
import shutil
from PIL import Image

# Create directory for style examples
STYLE_EXAMPLES_DIR = "style_examples"
os.makedirs(STYLE_EXAMPLES_DIR, exist_ok=True)

# Define styles from IDX_TO_CLASS
STYLES = [
    "Transitional",
    "Modern",
    "Minimalist",
    "Industrial",
    "Coastal",
    "Scandinavian",
    "Bohemian",
    "Mid-Century"
]

def setup_manual_examples():
    """
    Set up the directory structure for manual image placement.
    This function creates placeholder files that can be replaced with real images.
    """
    # Create the main directory if it doesn't exist
    if not os.path.exists(STYLE_EXAMPLES_DIR):
        os.makedirs(STYLE_EXAMPLES_DIR)
        print(f"Created directory: {STYLE_EXAMPLES_DIR}")
    
    # Create README with instructions
    readme_path = os.path.join(STYLE_EXAMPLES_DIR, "README.txt")
    with open(readme_path, "w") as f:
        f.write("Interior Design Style Examples\n")
        f.write("============================\n\n")
        f.write("For each style, place an image named [style_name]_example.jpg in this directory.\n")
        f.write("For example: modern_example.jpg, transitional_example.jpg, etc.\n\n")
        f.write("Required style examples:\n")
        for style in STYLES:
            f.write(f"- {style.lower()}_example.jpg\n")
    
    print(f"Created README with instructions at {readme_path}")
    
    # Create empty sample files that can be replaced
    for style in STYLES:
        file_path = os.path.join(STYLE_EXAMPLES_DIR, f"{style.lower()}_example.jpg")
        
        # If file already exists, don't touch it
        if os.path.exists(file_path):
            print(f"File already exists: {file_path}")
            continue
            
        # Create a placeholder 1x1 transparent image
        img = Image.new('RGB', (1, 1), color=(255, 255, 255))
        img.save(file_path)
        print(f"Created placeholder for {style.lower()}_example.jpg")
    
    print("\nDirectory setup complete!")
    print(f"\nTo add your images, place them in the {STYLE_EXAMPLES_DIR} directory with the correct naming:")
    for style in STYLES:
        print(f"- {style}: {style.lower()}_example.jpg")

def check_manual_examples():
    """Check which style examples are missing."""
    missing = []
    for style in STYLES:
        file_path = os.path.join(STYLE_EXAMPLES_DIR, f"{style.lower()}_example.jpg")
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1000:  # Less than 1KB is probably a placeholder
            missing.append(style)
    
    if missing:
        print("\nMissing or placeholder images for these styles:")
        for style in missing:
            print(f"- {style}")
    else:
        print("\nAll style images are present.")

if __name__ == "__main__":
    setup_manual_examples()
    check_manual_examples()
    print("\nRun this script again after adding images to check which ones are still missing.") 