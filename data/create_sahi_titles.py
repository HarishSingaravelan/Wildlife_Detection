import os
import json
import cv2
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_IMG_DIR = "/shared/rc/turbine"
JSON_DIR = "/shared/rc/turbine/turbine_split_stratified"
OUTPUT_DIR = "/shared/rc/turbine/turbine_sahi_tiles_1024"

TILE_SIZE = 1024
STRIDE = 800
TRUNC_THRESH = 0.50  # If a box is sliced, keep it only if 50% of the bird is visible
KEEP_BG_PROB = 0.05  # Keep exactly 5% of all purely empty background tiles
NUM_WORKERS = 30
# ==============================================================================

def process_single_image(img_info, annotations, split_name):
    """Slices a single massive image into 1024x1024 tiles."""
    in_path = os.path.join(INPUT_IMG_DIR, img_info['file_name'])
    
    img = cv2.imread(in_path)
    if img is None:
        return {'error': in_path}

    img_h, img_w = img.shape[:2]
    out_img_dir = os.path.join(OUTPUT_DIR, split_name, "images")
    os.makedirs(out_img_dir, exist_ok=True)

    x_steps = list(range(0, img_w - TILE_SIZE + 1, STRIDE))
    y_steps = list(range(0, img_h - TILE_SIZE + 1, STRIDE))
    
    if (img_w - TILE_SIZE) not in x_steps: x_steps.append(max(0, img_w - TILE_SIZE))
    if (img_h - TILE_SIZE) not in y_steps: y_steps.append(max(0, img_h - TILE_SIZE))

    generated_tiles = []
    generated_anns = []

    for y in y_steps:
        for x in x_steps:
            local_annotations = []
            has_bird = False
            
            for ann in annotations:
                bx, by, bw, bh = ann['bbox']
                bx2, by2 = bx + bw, by + bh
                
                ix1 = max(bx, x)
                iy1 = max(by, y)
                ix2 = min(bx2, x + TILE_SIZE)
                iy2 = min(by2, y + TILE_SIZE)
                
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    orig_area = bw * bh
                    
                    if (inter_area / orig_area) >= TRUNC_THRESH:
                        has_bird = True
                        nx, ny = ix1 - x, iy1 - y
                        nw, nh = ix2 - ix1, iy2 - iy1
                        
                        new_ann = ann.copy()
                        new_ann['bbox'] = [nx, ny, nw, nh]
                        new_ann['area'] = nw * nh
                        local_annotations.append(new_ann)

            if has_bird or random.random() < KEEP_BG_PROB:
                base_name = os.path.basename(img_info['file_name'])
                unique_id = img_info['id']
                
                # BUGFIX: Injected original image ID to guarantee globally unique filenames 
                tile_filename = f"img{unique_id}_{os.path.splitext(base_name)[0]}_x{x}_y{y}.jpg"
                out_path = os.path.join(out_img_dir, tile_filename)
                
                cropped_img = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                
                if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                    success = cv2.imwrite(out_path, cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not success:
                        print(f"⚠ FAILED TO WRITE IMAGE TO DISK: {out_path}")
                        continue
                        
                    tile_data = {
                        'file_name': f"images/{tile_filename}",
                        'width': cropped_img.shape[1],
                        'height': cropped_img.shape[0],
                        'original_image_id': img_info['id']
                    }
                    
                    generated_tiles.append(tile_data)
                    generated_anns.append(local_annotations)

    return {'tiles': generated_tiles, 'anns': generated_anns}

def unpack_and_process(task_args):
    return process_single_image(*task_args)

def process_split(split_name):
    input_json = os.path.join(JSON_DIR, split_name, "annotations.json")
    output_json = os.path.join(OUTPUT_DIR, split_name, "annotations.json")

    if not os.path.exists(input_json):
        return

    print(f"\n[INFO] Tiling Split: {split_name.upper()}")
    with open(input_json, 'r') as f:
        coco_data = json.load(f)

    ann_by_image = {img['id']: [] for img in coco_data['images']}
    for ann in coco_data['annotations']:
        ann_by_image[ann['image_id']].append(ann)

    new_images = []
    new_annotations = []
    missing_count = 0
    img_id_counter = 1
    ann_id_counter = 1

    tasks = [(img, ann_by_image[img['id']], split_name) for img in coco_data['images']]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(executor.map(unpack_and_process, tasks), total=len(tasks)))

    for res in results:
        if 'error' in res:
            missing_count += 1
            continue
            
        for tile_meta, local_anns in zip(res['tiles'], res['anns']):
            tile_meta['id'] = img_id_counter
            new_images.append(tile_meta)
            
            for ann in local_anns:
                ann['id'] = ann_id_counter
                ann['image_id'] = img_id_counter
                new_annotations.append(ann)
                ann_id_counter += 1
                
            img_id_counter += 1

    if missing_count > 0:
        print(f"⚠ Warning: {missing_count} images were missing.")

    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)
        
    print(f"✅ {split_name.upper()} successfully tiled! Created {len(new_images)} total tiles.")

def main():
    random.seed(42) 
    for split in ["train", "val", "test"]:
        process_split(split)
    print("\n🎉 SAHI TILING COMPLETE! Dataset is ready for production.")

if __name__ == "__main__":
    main()