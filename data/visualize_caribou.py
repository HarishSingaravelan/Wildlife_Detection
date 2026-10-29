import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ====================================================================
# CONFIGURATION
# ====================================================================
IMAGES_DIR = "../../../../shared/rc/turbine/Caribou/COCO_Dataset/images"
ANN_FILE = "../../../../shared/rc/turbine/caribou_split_stratified/train/annotations.json"
OUTPUT_DIR = "debug_visualizations"
NUM_IMAGES_TO_GENERATE = 5  # How many images to draw

# Colors for the different classes
COLORS = {
    1: 'red',    # Adult Caribou
    2: 'blue',   # Calf Caribou
    3: 'yellow'  # Ghost Caribou
}

def main():
    print(f"Loading annotations from: {ANN_FILE}")
    with open(ANN_FILE, 'r') as f:
        coco_data = json.load(f)

    # Create mapping dictionaries
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Map image IDs to their file names
    images_dict = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group annotations by image ID
    anns_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in anns_by_image:
            anns_by_image[img_id] = []
        anns_by_image[img_id].append(ann)

    # Filter out empty background images so we only look at images WITH caribou
    images_with_caribou = list(anns_by_image.keys())
    print(f"Total images with caribou: {len(images_with_caribou)}")

    if len(images_with_caribou) == 0:
        print("No caribou found in this dataset split!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pick random images to visualize
    sample_img_ids = random.sample(images_with_caribou, min(NUM_IMAGES_TO_GENERATE, len(images_with_caribou)))

    for idx, img_id in enumerate(sample_img_ids):
        filename = images_dict[img_id]
        img_path = os.path.join(IMAGES_DIR, filename)
        
        if not os.path.exists(img_path):
            print(f"⚠ Warning: Could not find image file {img_path}")
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(img)

        # Draw boxes
        for ann in anns_by_image[img_id]:
            cat_id = ann['category_id']
            cat_name = categories.get(cat_id, f"Class {cat_id}")
            
            # COCO format is [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            
            color = COLORS.get(cat_id, 'green')

            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # Add label text
            ax.text(x, y - 5, f"{cat_name}", color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1))

        # Formatting
        plt.axis('off')
        plt.title(f"Image: {filename}")
        plt.tight_layout()

        # Save to disk
        out_path = os.path.join(OUTPUT_DIR, f"debug_{idx}_{filename}")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization to {out_path}")

if __name__ == "__main__":
    main()