"""
DETR Model with Existing Faster R-CNN Data Pipeline - BULLETPROOF EDITION
"""

import torch
import torch.nn as nn
from transformers import (
    DetrForObjectDetection, 
    DetrImageProcessor,
    DFineForObjectDetection,
    AutoImageProcessor,
    DeformableDetrImageProcessor,
    DeformableDetrForObjectDetection
)
import transformers.models.detr.modeling_detr as modeling_detr
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple
import os
import math
import torch.nn.functional as F
from tqdm.auto import tqdm

# ==============================================================================
# FASTER R-CNN STATIC PATCH (Tuple-Safe)
# ==============================================================================
from torchvision.models.detection import roi_heads
orig_fastrcnn_loss = roi_heads.fastrcnn_loss

GLOBAL_FASTRCNN_LOGS = {}

def patched_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    global GLOBAL_FASTRCNN_LOGS
    losses = orig_fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    
    with torch.no_grad():
        cat_labels = torch.cat(labels, dim=0)
        unreduced_loss = F.cross_entropy(class_logits, cat_labels, reduction="none")
        
        GLOBAL_FASTRCNN_LOGS.clear()
        for class_id in cat_labels.unique():
            c_id = class_id.item()
            GLOBAL_FASTRCNN_LOGS[f"log_only_class_{c_id}"] = unreduced_loss[cat_labels == c_id].mean()
            
    return losses

roi_heads.fastrcnn_loss = patched_fastrcnn_loss

# ==============================================================================
# TRANSFORM ADAPTER & PROCESSOR WRAPPER
# ==============================================================================
class DETRTransformAdapter:
    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform
    
    def __call__(self, image, target):
        # 1. Run Albumentations
        image, target = self.albumentations_transform(image, target)
        
        # 2. To Numpy HWC
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:
                image_np = image.permute(1, 2, 0).numpy()
            else:
                image_np = image.numpy()
        else:
            image_np = np.array(image)

        # 3. Clean up the image
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=255.0, neginf=0.0)
        if image_np.dtype != np.uint8:
            if image_np.max() <= 2.0: 
                image_np = image_np * 255.0
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        orig_h, orig_w = image_np.shape[:2]
        
        # 4. Prepare pure COCO annotations for the HF Processor
        boxes = target["boxes"].cpu().numpy() if isinstance(target["boxes"], torch.Tensor) else target["boxes"]
        labels = target["labels"].cpu().numpy() if isinstance(target["labels"], torch.Tensor) else target["labels"]
        
        valid_annotations = []
        for i in range(len(boxes)):
            if np.any(np.isnan(boxes[i])):
                continue
                
            x1, y1, x2, y2 = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
            
            w = x2 - x1
            h = y2 - y1
            
            if w >= 1.0 and h >= 1.0:
                valid_annotations.append({
                    "bbox": [x1, y1, w, h], 
                    "category_id": int(labels[i]),
                    "area": float(w * h),
                    "iscrowd": 0,
                })

        img_id = target.get("image_id", 0)
        img_id = img_id.item() if isinstance(img_id, torch.Tensor) else img_id
        
        annotations = {
            "image_id": img_id,
            "annotations": valid_annotations
        }
        
        # Hand off the raw components. They will be processed as a unified batch below.
        return image_np, annotations


class DETRCollateWrapper:
    """ This safely wraps Hugging Face's processing engine so PyTorch DataLoader can use it """
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch):
        images = [item[0] for item in batch]
        annotations = [item[1] for item in batch]
        
        # Hugging Face calculates all the padding, masks, and normalization correctly!
        processed = self.processor(images=images, annotations=annotations, return_tensors="pt")
        
        pixel_values = processed["pixel_values"]
        
        # Securely extract or generate the attention mask
        if "pixel_mask" in processed:
            pixel_mask = processed["pixel_mask"]
        else:
            pixel_mask = torch.ones((pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3]), dtype=torch.long)
            
        return pixel_values, pixel_mask, processed["labels"]


# ==============================================================================
# PIPELINE ARCHITECTURE
# ==============================================================================
class DETRWithExistingDataPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['training']['device'])
        
        arch = config['model'].get('architecture', 'standard_detr')
        model_name = config['model'].get('pretrained_model', 'facebook/detr-resnet-50')
        print(f"\n[INFO] Initializing Architecture: {arch.upper()} from {model_name}")

        class_names = config['model']['class_names']
        id2label = {i: name for i, name in enumerate(class_names)}
        label2id = {name: i for i, name in enumerate(class_names)}

        if arch == 'dfine':
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = DFineForObjectDetection.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
            self.model.to(self.device)
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config['training']['learning_rate'] * 0.1}, 
            ]
            
        elif arch == 'deformable_detr':
            self.processor = DeformableDetrImageProcessor.from_pretrained(model_name, do_convert_annotations=True)
            self.model = DeformableDetrForObjectDetection.from_pretrained(model_name, num_labels=config['model']['num_object_classes'], ignore_mismatched_sizes=True)
            self.model.to(self.device)
            
            # --- MAGIC FIX: Prevent NaN Explosion in Classification Head ---
            print("  ↳ Applying Prior Probability initialization to Deformable DETR classification head...")
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            for module in self.model.class_embed:
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.constant_(module.bias, bias_value)
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.normal_(module.weight, std=0.01)
                    
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {"params": [p for n, p in self.model.named_parameters() if "backbone" in n and p.requires_grad], "lr": config['training']['learning_rate'] * 0.1}, 
            ]
            
        elif arch == 'faster_rcnn':
            print("  -> Bypassing HuggingFace model load (Child class will build Faster R-CNN)")
            self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', do_convert_annotations=True)
            self.model = None  
            param_dicts = []   
            
        else:
            self.processor = DetrImageProcessor.from_pretrained(model_name, do_convert_annotations=True)
            self.model = DetrForObjectDetection.from_pretrained(model_name, num_labels=config['model']['num_object_classes'], ignore_mismatched_sizes=True)
            self.model.to(self.device)
            param_dicts = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad]}
            ]

        import sys
        from pathlib import Path
        ROOT_DIR = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(ROOT_DIR))

        from turbine_processing.dataset import TurbineCocoDataset
        from turbine_processing.transforms_detr import get_train_transform_detr_medium, get_val_transform_detr

        train_transform = DETRTransformAdapter(get_train_transform_detr_medium())
        val_transform = DETRTransformAdapter(get_val_transform_detr())
        
        self.train_dataset = TurbineCocoDataset(images_dir=config['data']['train_images_dir'], ann_file=config['data']['train_ann_file'], transforms=train_transform)
        self.val_dataset = TurbineCocoDataset(images_dir=config['data']['val_images_dir'], ann_file=config['data']['val_ann_file'], transforms=val_transform)
        self.test_dataset = TurbineCocoDataset(images_dir=config['data']['test_images_dir'], ann_file=config['data']['test_ann_file'], transforms=val_transform)
        
        # THE FIX: Properly assign the collate_fn so train_detr.py can find it!
        self.collate_fn = DETRCollateWrapper(self.processor)

        loader_kwargs = {
            "batch_size": config['training']['batch_size'],
            "num_workers": config['training']['num_workers'],
            "collate_fn": self.collate_fn,
            "pin_memory": True,
            "persistent_workers": True if config['training']['num_workers'] > 0 else False,
            "prefetch_factor": 8 if config['training']['num_workers'] > 0 else None
        }

        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_kwargs)
        
        self.sampler = None
        self.use_adaptive_sampler = config['data'].get('use_adaptive_sampler', False)
        
        if self.use_adaptive_sampler:
            from turbine_processing.sampler_adaptive import AdaptiveDETRSampler
            self.sampler = AdaptiveDETRSampler(
                dataset=self.train_dataset,
                epoch_size=len(self.train_dataset),
                initial_mode=config['data'].get('initial_mode', 'equal'),
                adaptation_rate=config['data'].get('adaptation_rate', 0.3),
                min_weight=config['data'].get('min_weight', 0.1),
                max_weight=config['data'].get('max_weight', 5.0),
                background_ratio=config['data'].get('background_ratio', 0.5),
                dynamic_background=config['data'].get('dynamic_background', False),
                min_bg_ratio=config['data'].get('min_bg_ratio', 0.15),
                max_bg_ratio=config['data'].get('max_bg_ratio', 0.5),
            )
            self.train_loader = DataLoader(self.train_dataset, shuffle=False, sampler=self.sampler, **loader_kwargs)
            print("\n✓ USING ADAPTIVE SAMPLER")
        else:
            self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_kwargs)
            print("\n✓ USING REGULAR SHUFFLING (no sampler)")
        
        if len(param_dicts) > 0:
            self.optimizer = torch.optim.AdamW(param_dicts, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config['training'].get('scheduler_step_size', 30), gamma=config['training'].get('scheduler_gamma', 0.1))
        else:
            self.optimizer = None
            self.scheduler = None
            print("  -> Bypassing Base Optimizer (Child class will build it)")
        
        self.best_map = 0.0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.eval_every_n_epochs = config['evaluation'].get('eval_every_n_epochs', 5)