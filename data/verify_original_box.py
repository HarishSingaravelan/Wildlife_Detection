import os
import json
import cv2

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Point this to the ORIGINAL dataset JSON (before SAHI)
ORIGINAL_JSON_PATH = "/shared/rc/turbine/turbine_split_stratified/train/annotations.json"
ORIGINAL_IMG_DIR = "/shared/rc/turbine"
OUTPUT_DIR = "./debug_original_annotations"

# The file you want to investigate (can be partial name like 'DJI_0805')
TARGET_FILENAME = "DJI_0367" 
# ==============================================================================

def verify_original_annotation():
    print(f"Loading original annotations from {ORIGINAL_JSON_PATH}...")
    with open(ORIGINAL_JSON_PATH, 'r') as f:
        coco_data = json.load(f)

    cat_dict = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}

    # Find all images that match the target filename
    matching_images = [img for img in coco_data['images'] if TARGET_FILENAME in img['file_name']]
    
    if not matching_images:
        print(f"❌ Could not find any image matching '{TARGET_FILENAME}' in the JSON.")
        return

    print(f"Found {len(matching_images)} matching image(s). Processing...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use enumerate to keep a running counter (1, 2, 3...) for the images
    for idx, img_info in enumerate(matching_images, start=1):
        img_path = os.path.join(ORIGINAL_IMG_DIR, img_info['file_name'])
        print(f"Loading {img_path}...")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠ Could not read image file from disk: {img_path}")
            continue

        # Get all annotations for this specific image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']]
        
        if not annotations:
            print(f"Image {img_info['file_name']} has NO annotations in the JSON.")
            continue

        # Draw the original boxes
        for ann in annotations:
            x, y, w, h = ann['bbox']
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            cat_id = ann['category_id']
            class_name = cat_dict.get(cat_id, str(cat_id))

            # Draw a VERY THICK red box (thickness=8) so it's visible on a 5000x4000 image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 8)
            
            # Draw large text
            cv2.rectangle(img, (x1, y1 - 60), (x1 + 400, y1), (0, 0, 255), -1)
            cv2.putText(img, class_name, (x1 + 10, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

        # Split the filename to inject the counter before the extension
        base_name = os.path.basename(img_info['file_name'])
        name_only, ext = os.path.splitext(base_name)
        
        # Format the new filename with a 2-digit zero-padded index (01, 02, etc.)
        numbered_filename = f"VERIFIED_{name_only}_{idx:02d}{ext}"
        out_path = os.path.join(OUTPUT_DIR, numbered_filename)
        
        cv2.imwrite(out_path, img)
        print(f"✅ Saved verification image to: {out_path}")

if __name__ == "__main__":
    verify_original_annotation()