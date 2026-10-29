import os
import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Allow PIL to open massive TIFF images without crashing
Image.MAX_IMAGE_PIXELS = None

class CaribouPreprocessor:
    def __init__(self, input_image_dir, input_csv_dir, output_dir, 
                 tile_size=1024, stride=800, trunc_thresh=0.6, 
                 target_bg_ratio=0.50, keep_all_backgrounds=False):
        
        self.input_image_dir = Path(input_image_dir)
        self.input_csv_dir = Path(input_csv_dir)
        self.output_dir = Path(output_dir)
        self.images_out_dir = self.output_dir / "images"
        
        # Methodology Hyperparameters
        self.tile_size = tile_size
        self.stride = stride
        self.trunc_thresh = trunc_thresh       
        self.target_bg_ratio = target_bg_ratio 
        self.keep_all_backgrounds = keep_all_backgrounds # The 100% override switch
        self.aspect_ratio_assumption = 0.75    

        # COCO Format Skeleton
        self.coco = {
            "info": {"description": "Caribou Tiled Orthomosaics", "version": "1.0"},
            "images": [],
            "annotations": [],
            "categories": []
        }
        self.category_map = {} 
        self.annot_id_counter = 1

    def _parse_custom_csv(self, csv_path):
        """Parses the custom CSV, extracting metadata and applying dynamic 2D padding."""
        filename, label_id, label_name = None, None, None
        boxes = []
        
        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#filename:'):
                    filename = line.split(':', 1)[1].strip()
                elif line.startswith('#label_id:'):
                    label_id = int(line.split(':', 1)[1].strip())
                elif line.startswith('#label:'):
                    label_name = line.split(':', 1)[1].strip()
                elif not line.startswith('#') and line and line != 'x1,y1,x2,y2':
                    x1, y1, x2, y2 = map(float, line.split(','))
                    
                    # --- DYNAMIC ASPECT RATIO PADDING ---
                    width = x2 - x1
                    if y1 == y2:
                        estimated_height = width * self.aspect_ratio_assumption
                        
                        # NUDGE VALUES: Adjust these until the visualizer shows the boxes on the bodies!
                        # Positive X moves right, Positive Y moves down.
                        x_nudge = 10.0  
                        y_nudge = 15.0  
                        
                        x1 = x1 + x_nudge
                        x2 = x2 + x_nudge
                        y1 = (y1 - (estimated_height / 2.0)) + y_nudge
                        y2 = (y2 + (estimated_height / 2.0)) + y_nudge
                    
                    boxes.append([x1, y1, x2, y2])
        
        return filename, label_id, label_name, boxes

    def process_dataset(self):
        self.images_out_dir.mkdir(parents=True, exist_ok=True)
        
        csv_files = list(self.input_csv_dir.glob("*.csv"))
        print(f"🔍 Found {len(csv_files)} annotation CSVs.")

        positive_tiles_saved = 0
        background_candidates = [] 

        # Main progress bar for CSVs
        for csv_path in tqdm(csv_files, desc="Processing Mosaics"):
            filename, label_id, label_name, global_boxes = self._parse_custom_csv(csv_path)
            
            if filename is None:
                continue

            # Ensure Category exists in COCO
            if label_id not in self.category_map:
                self.category_map[label_id] = label_id
                self.coco["categories"].append({"id": label_id, "name": label_name})

            # --- BULLETPROOF EXTENSION CHECK ---
            img_path = self.input_image_dir / filename
            if not img_path.exists():
                if img_path.suffix == '.tif':
                    img_path = img_path.with_suffix('.tiff')
                elif img_path.suffix == '.tiff':
                    img_path = img_path.with_suffix('.tif')
                    
            if not img_path.exists():
                print(f"\n⚠ Warning: Image {filename} not found. Skipping.")
                continue

            # Load massive TIFF into memory
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Generate Sliding Window Coordinates
            x_steps = list(range(0, img_w - self.tile_size + 1, self.stride))
            y_steps = list(range(0, img_h - self.tile_size + 1, self.stride))
            
            # Sub-progress bar for slicing the massive image
            total_tiles = len(x_steps) * len(y_steps)
            with tqdm(total=total_tiles, desc=f"Slicing {img_path.name[:15]}...", leave=False) as pbar:
                for y in y_steps:
                    for x in x_steps:
                        tile_has_caribou = False
                        local_annotations = []
                        
                        # Check which global boxes fall into this tile
                        for box in global_boxes:
                            bx1, by1, bx2, by2 = box
                            
                            # Calculate intersection
                            ix1 = max(bx1, x)
                            iy1 = max(by1, y)
                            ix2 = min(bx2, x + self.tile_size)
                            iy2 = min(by2, y + self.tile_size)
                            
                            if ix1 < ix2 and iy1 < iy2:
                                inter_area = (ix2 - ix1) * (iy2 - iy1)
                                orig_area = (bx2 - bx1) * (by2 - by1)
                                
                                # --- TRUNCATION RULE (60%) ---
                                if (inter_area / orig_area) >= self.trunc_thresh:
                                    tile_has_caribou = True
                                    nx1 = ix1 - x
                                    ny1 = iy1 - y
                                    nx2 = ix2 - x
                                    ny2 = iy2 - y
                                    
                                    local_annotations.append({
                                        "category_id": label_id,
                                        "bbox": [nx1, ny1, nx2 - nx1, ny2 - ny1], 
                                        "area": (nx2 - nx1) * (ny2 - ny1)
                                    })
                        
                        tile_filename = f"{img_path.stem}_{x}_{y}.jpg"
                        tile_data = {
                            "img_obj": img, "x": x, "y": y, 
                            "filename": tile_filename, "annotations": local_annotations
                        }
                        
                        if tile_has_caribou:
                            self._save_tile_and_coco(tile_data)
                            positive_tiles_saved += 1
                        else:
                            background_candidates.append(tile_data)
                        
                        pbar.update(1)

        # --- BACKGROUND FILTERING LOGIC ---
        print(f"\n📊 Extracted {positive_tiles_saved} tiles containing Caribou.")
        
        if self.keep_all_backgrounds:
            print(f"⚠ Override Active: Saving ALL {len(background_candidates)} empty background tiles.")
            selected_bgs = background_candidates
        else:
            # Safe math to prevent zero division
            safe_ratio = min(self.target_bg_ratio, 0.99) 
            num_bg_to_keep = int(positive_tiles_saved * (safe_ratio / (1.0 - safe_ratio)))
            num_bg_to_keep = min(num_bg_to_keep, len(background_candidates))
            print(f"🎲 Randomly sampling {num_bg_to_keep} empty background tiles (Target Ratio: {safe_ratio*100}%).")
            selected_bgs = random.sample(background_candidates, num_bg_to_keep)

        for bg_tile in tqdm(selected_bgs, desc="Saving Background Tiles"):
            self._save_tile_and_coco(bg_tile)

        # Write final COCO JSON
        out_json = self.output_dir / "caribou_coco_annotations.json"
        with open(out_json, "w") as f:
            json.dump(self.coco, f)
        
        print(f"\n✅ Processing Complete! Saved to {self.output_dir}")
        print(f"Total Images in COCO: {len(self.coco['images'])}")
        print(f"Total Annotations in COCO: {len(self.coco['annotations'])}")

    def _save_tile_and_coco(self, tile_data):
        """Crops the image, saves the JPG, and updates the COCO dict."""
        x, y = tile_data["x"], tile_data["y"]
        img_crop = tile_data["img_obj"].crop((x, y, x + self.tile_size, y + self.tile_size))
        
        out_path = self.images_out_dir / tile_data["filename"]
        if img_crop.mode in ("RGBA", "P"):
            img_crop = img_crop.convert("RGB")
        img_crop.save(out_path, format="JPEG", quality=95)
        
        image_id = len(self.coco["images"]) + 1
        self.coco["images"].append({
            "id": image_id,
            "file_name": tile_data["filename"],
            "width": self.tile_size,
            "height": self.tile_size
        })
        
        for ann in tile_data["annotations"]:
            self.coco["annotations"].append({
                "id": self.annot_id_counter,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["area"],
                "iscrowd": 0
            })
            self.annot_id_counter += 1


if __name__ == "__main__":
    # =========================================================
    # USER CONFIGURATION
    # =========================================================
    INPUT_IMAGE_DIR = "/shared/rc/turbine/Caribou/tif"  # Folder containing .tif files
    INPUT_CSV_DIR   = "/shared/rc/turbine/Caribou/csvs_fixed" # Folder containing .csv files      
    OUTPUT_DIR      = "/shared/rc/turbine/Caribou/COCO_Dataset" # Where to save the final dataset

    processor = CaribouPreprocessor(
        input_image_dir=INPUT_IMAGE_DIR,
        input_csv_dir=INPUT_CSV_DIR,
        output_dir=OUTPUT_DIR,
        tile_size=1024,
        stride=800,
        trunc_thresh=0.60,
        
        # --- BACKGROUND CONTROLS ---
        # Set keep_all_backgrounds=True to save 100% of the empty tiles.
        # Set it to False if you want to use the ratio formula instead.
        keep_all_backgrounds=True, 
        target_bg_ratio=0.50       
    )
    
    processor.process_dataset()