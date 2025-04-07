# prompt_engineering.py

def generate_prompt(current_style: str, target_style: str) -> str:
    """
    Generate a prompt to transform the current style into the target style.
    Example:
      "Transform a transitional interior into a modern style interior with modern elements, ample lighting, and elegant decor."
    """
    style_attributes = {
        "modern": "clean lines, minimal decoration, neutral colors with occasional bold accents, open spaces, and sleek furniture",
        "transitional": "blend of traditional and contemporary elements, neutral color palette, comfortable and practical furniture, subtle patterns and textures",
        "industrial": "raw materials like exposed brick and metal, utilitarian objects, vintage or reclaimed items, neutral colors with emphasis on gray and brown",
        "coastal": "light and airy color palette, natural light, beach-inspired textures and accessories, comfortable casual furniture",
        "minimalist": "extreme simplicity, limited color palette (usually monochromatic), clean lines, functional furniture, lack of clutter",
        "scandinavian": "simple, functional design with light woods, white walls, clean lines, cozy textiles, and natural materials",
        "bohemian": "rich patterns, textures, and colors, vintage furniture, plants, global influences, layered textiles, and eclectic decor",
        "mid-century": "organic forms, clean lines, minimal ornamentation, mix of traditional and non-traditional materials, graphic patterns, and iconic furniture pieces"
    }
    
    current_attr = style_attributes.get(current_style.lower(), "distinctive elements")
    target_attr = style_attributes.get(target_style.lower(), "distinctive elements")
    
    prompt = (
        f"Transform a {current_style.lower()} interior with {current_attr} into a {target_style.lower()} "
        f"style interior featuring {target_attr}. Maintain the original layout but update all design elements "
        f"to reflect the {target_style.lower()} aesthetic."
    )
    
    return prompt

# Test the prompt generation when the script is run directly
if __name__ == "__main__":
    # Test with some example style transformations
    test_cases = [
        ("Transitional", "Modern"),
        ("Modern", "Transitional"),
        ("Industrial", "Coastal"),
        ("Minimalist", "Industrial")
    ]
    
    print("Generated Prompts for Style Transformations:")
    print("-" * 80)
    
    for current, target in test_cases:
        prompt = generate_prompt(current, target)
        print(f"\n{current} → {target}:")
        print(prompt)
        print("-" * 80)