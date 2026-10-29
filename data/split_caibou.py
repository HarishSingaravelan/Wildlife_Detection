"""
Stratified Dataset Splitter for Caribou Tiled Orthomosaics
Optimized for COCO-formatted object detection tracking.
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List

def load_coco_annotations(ann_file: Path) -> Dict:
    print(f"Loading annotations from: {ann_file}")
    with open(ann_file, 'r') as f:
        return json.load(f)

def aggregate_rare_classes(coco_data: Dict, min_images: int = 3, target_bucket_name: str = "unknown") -> Dict:
    """Groups ultra-rare items into a safe common category so splitting is mathematically possible."""
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    if not annotations or not categories:
        print("ℹ No annotations or categories found to aggregate. Skipping grouping phase.")
        return coco_data

    print(f"\n[1/6] Aggregating classes with < {min_images} images into '{target_bucket_name}'...")
    
    img_counts_per_class = defaultdict(set)
    for ann in annotations:
        img_counts_per_class[ann['category_id']].add(ann['image_id'])
        
    cat_id_to_name = {c['id']: c['name'] for c in categories}
    
    rare_cat_ids = set()
    rare_details = []
    for cat_id, img_set in img_counts_per_class.items():
        if len(img_set) < min_images and cat_id_to_name.get(cat_id) != target_bucket_name:
            rare_cat_ids.add(cat_id)
            rare_details.append((cat_id_to_name.get(cat_id, "Unnamed"), len(img_set)))
            
    if not rare_cat_ids:
        print("  ✓ All classes meet criteria. No bucketing required.")
        return coco_data

    target_cat_id = None
    for c in categories:
        if c['name'] == target_bucket_name:
            target_cat_id = c['id']
            break
            
    if target_cat_id is None:
        target_cat_id = max([c['id'] for c in categories] + [0]) + 1
        categories.append({"id": target_cat_id, "name": target_bucket_name, "supercategory": "none"})
        print(f"  Created category '{target_bucket_name}' with ID {target_cat_id}")

    print(f"  Re-mapping {len(rare_cat_ids)} rare classes:")
    for name, count in rare_details:
        print(f"    - {name} ({count} images) -> {target_bucket_name}")
        
    for ann in annotations:
        if ann['category_id'] in rare_cat_ids:
            ann['category_id'] = target_cat_id

    new_categories = [c for c in categories if c['id'] not in rare_cat_ids]
    print(f"  Surviving Unique Classes: {len(new_categories)} / {len(coco_data['categories'])}")
    
    coco_data['categories'] = new_categories
    return coco_data

def stratified_split(coco_data: Dict, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """Splits images using Rarest-First bucket stratification."""
    random.seed(seed)
    
    annotations = coco_data.get('annotations', [])
    images = coco_data.get('images', [])
    
    img_counts_per_class = defaultdict(set)
    for ann in annotations:
        img_counts_per_class[ann['category_id']].add(ann['image_id'])
        
    global_freq = {cat_id: len(imgs) for cat_id, imgs in img_counts_per_class.items()}
    buckets = defaultdict(list)
    
    # Map annotations by image for fast indexing
    img_to_anns = defaultdict(list)
    for a in annotations:
        img_to_anns[a['image_id']].append(a)
    
    for img in images:
        img_id = img['id']
        img_anns = img_to_anns[img_id]
        
        if not img_anns:
            buckets['background'].append(img_id)
        else:
            rarest_cat = min(img_anns, key=lambda a: global_freq.get(a['category_id'], 0))['category_id']
            buckets[rarest_cat].append(img_id)
            
    train_ids, val_ids, test_ids = [], [], []
    sorted_cats = sorted(global_freq.keys(), key=lambda k: global_freq[k])
    
    for cat_id in sorted_cats:
        bucket_imgs = buckets[cat_id]
        random.shuffle(bucket_imgs)
        
        n = len(bucket_imgs)
        if n == 0:
            continue
        elif n == 1:
            train_ids.append(bucket_imgs[0])
        elif n == 2:
            train_ids.extend([bucket_imgs[0]])
            val_ids.extend([bucket_imgs[1]])
        else:
            train_ids.append(bucket_imgs[0])
            val_ids.append(bucket_imgs[1])
            test_ids.append(bucket_imgs[2])
            
            rest = bucket_imgs[3:]
            if rest:
                n_rest = len(rest)
                n_train = int(n_rest * train_ratio)
                n_val = int(n_rest * val_ratio)
                
                train_ids.extend(rest[:n_train])
                val_ids.extend(rest[n_train:n_train + n_val])
                test_ids.extend(rest[n_train + n_val:])
                
    bg_imgs = buckets['background']
    random.shuffle(bg_imgs)
    n_bg = len(bg_imgs)
    n_train_bg = int(n_bg * train_ratio)
    n_val_bg = int(n_bg * val_ratio)
    
    train_ids.extend(bg_imgs[:n_train_bg])
    val_ids.extend(bg_imgs[n_train_bg:n_train_bg + n_val_bg])
    test_ids.extend(bg_imgs[n_train_bg + n_val_bg:])
    
    return train_ids, val_ids, test_ids

def create_split_annotations(coco_data: Dict, split_image_ids: List[int]) -> Dict:
    split_image_ids_set = set(split_image_ids)
    return {
        'images': [img for img in coco_data.get('images', []) if img['id'] in split_image_ids_set],
        'annotations': [ann for ann in coco_data.get('annotations', []) if ann['image_id'] in split_image_ids_set],
        'categories': coco_data.get('categories', []),
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', [])
    }

def copy_images(source_root: Path, dest_root: Path, image_filenames: List[str]):
    dest_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for img_file in image_filenames:
        src_path = source_root / img_file
        dest_path = dest_root / img_file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            copied += 1
    print(f"  Copied: {copied} / {len(image_filenames)} images")

def verify_distribution(train_coco, val_coco, test_coco):
    print("\n[6/6] Verifying Split Quality (Class Coverage Report)...")
    train_cats = set(ann['category_id'] for ann in train_coco.get('annotations', []))
    val_cats = set(ann['category_id'] for ann in val_coco.get('annotations', []))
    test_cats = set(ann['category_id'] for ann in test_coco.get('annotations', []))
    
    all_cats = set(c['id'] for c in train_coco.get('categories', []))
    
    missing_in_val = all_cats - val_cats
    missing_in_test = all_cats - test_cats
    
    if not missing_in_val and not missing_in_test:
        print("  SUCCESS: Every single class is perfectly represented in Train, Val, and Test splits!")
    else:
        print(f"  ⚠ Distribution Warnings:")
        if missing_in_val: print(f"    - Missing in Val split: {len(missing_in_val)} classes")
        if missing_in_test: print(f"    - Missing in Test split: {len(missing_in_test)} classes")

def main():
    parser = argparse.ArgumentParser(description='Stratified Splitter for Caribou Dataset')
    parser.add_argument('--source_root', type=str, required=True, help='Path containing original images')
    parser.add_argument('--annotation_file', type=str, required=True, help='COCO master JSON file')
    parser.add_argument('--output_root', type=str, required=True, help='Target directory for output sets')
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--min_images', type=int, default=3)
    parser.add_argument('--bucket_name', type=str, default="unknown")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy_images', action='store_true')
    
    args = parser.parse_args()
    source_root = Path(args.source_root)
    annotation_file = Path(args.annotation_file)
    output_root = Path(args.output_root)
    
    print("="*60)
    print("RUNNING STRATIFIED SPLIT FOR CARIBOU ORTHOMOSAICS")
    print("="*60)
    
    coco_data = load_coco_annotations(annotation_file)
    coco_data = aggregate_rare_classes(coco_data, args.min_images, args.bucket_name)
    
    print("\n[2/6] Executing Rarest-First Stratification...")
    train_ids, val_ids, test_ids = stratified_split(
        coco_data, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"  Split Summary -> Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)} images")
    
    print("\n[3/6] Mapping annotations to subset splits...")
    train_coco = create_split_annotations(coco_data, train_ids)
    val_coco = create_split_annotations(coco_data, val_ids)
    test_coco = create_split_annotations(coco_data, test_ids)
    
    print("\n[4/6] Exporting split JSON annotation files...")
    output_root.mkdir(parents=True, exist_ok=True)
    for name, data in zip(['train', 'val', 'test'], [train_coco, val_coco, test_coco]):
        path = output_root / name / 'annotations.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"   Exported: {path}")
        
    if args.copy_images:
        print("\n[5/6] Transporting image profiles across directories...")
        id_to_file = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
        copy_images(source_root, output_root / 'train' / 'images', [id_to_file[i] for i in train_ids])
        copy_images(source_root, output_root / 'val' / 'images', [id_to_file[i] for i in val_ids])
        copy_images(source_root, output_root / 'test' / 'images', [id_to_file[i] for i in test_ids])
    else:
        print("\n[5/6] Skipping physical image copying (--copy_images flag unset)")
        
    verify_distribution(train_coco, val_coco, test_coco)
    print("\n" + "="*60 + "\n DATASET SPLIT COMPLETE!\n" + "="*60)

if __name__ == '__main__':
    main()