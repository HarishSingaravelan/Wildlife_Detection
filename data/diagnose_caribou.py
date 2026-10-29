"""
Diagnostic script for Caribou COCO annotations.
Scans the dataset and generates the exact 'model:' block needed for config.yaml.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def analyze_annotations(ann_file: str):
    print(f"\n{'='*60}")
    print(f"Analyzing: {ann_file}")
    print(f"{'='*60}")

    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    num_images = len(data.get('images', []))
    print(f"\n IMAGES:")
    print(f"   Total images: {num_images}")

    # ── Categories ────────────────────────────────────────────────
    print("\n CATEGORIES:")
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    for cat_id, cat_name in sorted(categories.items()):
        print(f"   id={cat_id}  name={cat_name}")

    cat_ids = sorted(categories.keys())
    print(f"\n   Min category_id : {min(cat_ids)}")
    print(f"   Max category_id : {max(cat_ids)}")
    print(f"   Num categories  : {len(cat_ids)}")

    # ── Annotations ───────────────────────────────────────────────
    annotations = data.get('annotations', [])
    print(f"\n ANNOTATIONS:")
    print(f"   Total annotations: {len(annotations)}")

    cat_counts = defaultdict(int)
    for ann in annotations:
        cat_counts[ann['category_id']] += 1

    print("\n   Annotations per category:")
    for cat_id in sorted(cat_counts.keys()):
        name = categories.get(cat_id, "UNKNOWN!")
        print(f"   id={cat_id}  name={name:30s}  count={cat_counts[cat_id]}")

    return categories, cat_ids

def generate_yaml_config(categories: dict, cat_ids: list):
    """Generates the exact YAML block needed for training."""
    print(f"\n{'='*60}")
    print("🚀 GENERATED CONFIGURATION FOR config.yaml")
    print(f"{'='*60}")
    
    min_id = min(cat_ids)
    max_id = max(cat_ids)
    
    if min_id == 1:
        num_classes = max_id + 1
        print("Note: Dataset is 1-indexed. Adding a padding 'background' class at index 0.\n")
    else:
        num_classes = max_id + 1
        print("Note: Dataset is 0-indexed.\n")

    print(f"model:\n  num_object_classes: {num_classes}")
    print("  class_names:")
    
    if min_id == 1:
        print('   - "background"                    # index 0')
        
    for cat_id in range(min_id, max_id + 1):
        name = categories.get(cat_id, f"unused_class_{cat_id}")
        print(f'   - "{name}"{" "*(30-len(name))}# index {cat_id}')

if __name__ == "__main__":
    # Pointing directly to your newly created Caribou splits
    base_dir = Path("../../../../shared/rc/turbine/caribou_split_stratified")
    
    ann_files = [
        base_dir / "train" / "annotations.json",
        base_dir / "val" / "annotations.json",
        base_dir / "test" / "annotations.json",
    ]

    master_categories = {}
    master_cat_ids = set()

    for ann_file in ann_files:
        if ann_file.exists():
            cats, ids = analyze_annotations(str(ann_file))
            master_categories.update(cats)
            master_cat_ids.update(ids)
        else:
            print(f"\n⚠ File not found: {ann_file}")

    if master_categories:
        generate_yaml_config(master_categories, list(master_cat_ids))