"""
Dataset Re-mapper for Caribou Dataset
Condenses category IDs 20, 21, 22 down to 1, 2, 3 to optimize tracking and sampling.
"""

import json
from pathlib import Path

def remap_caribou_splits(base_dir: Path):
    splits = ['train', 'val', 'test']
    
    # 1. Define the exact translation map
    id_mapping = {
        20: 1,  # Adult Caribou
        21: 2,  # Calf Caribou
        22: 3   # Ghost Caribou
    }
    
    print(f"{'='*60}")
    print("REMAPPING CARIBOU CATEGORY IDs")
    print(f"{'='*60}")

    for split in splits:
        ann_path = base_dir / split / 'annotations.json'
        if not ann_path.exists():
            print(f"⚠ Skipping {split}: File not found at {ann_path}")
            continue
            
        print(f"\nProcessing: {split.upper()} split...")
        with open(ann_path, 'r') as f:
            data = json.load(f)
            
        # 2. Update Annotations
        updated_anns = 0
        for ann in data.get('annotations', []):
            old_id = ann['category_id']
            if old_id in id_mapping:
                ann['category_id'] = id_mapping[old_id]
                updated_anns += 1
                
        # 3. Update Categories
        new_categories = []
        for cat in data.get('categories', []):
            old_id = cat['id']
            if old_id in id_mapping:
                cat['id'] = id_mapping[old_id]
                new_categories.append(cat)
                
        # Update the JSON dictionary with the cleaned categories
        data['categories'] = sorted(new_categories, key=lambda x: x['id'])
        
        # 4. Save the file back (Overwriting the old IDs)
        with open(ann_path, 'w') as f:
            json.dump(data, f)
            
        print(f"  ✓ Remapped {updated_anns} bounding boxes.")
        print(f"  ✓ Categories are now: {[c['id'] for c in data['categories']]}")

if __name__ == "__main__":
    # Point this directly to your stratified split folder
    dataset_dir = Path("../../../../shared/rc/turbine/caribou_split_stratified")
    remap_caribou_splits(dataset_dir)
    print("\nDataset is now fully compacted. Ready for training!")