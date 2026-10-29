"""
Transforms for DETR / Deformable DETR / D-FINE
Engineered for Absolute Pixels -> Strict Area Clamping -> HF Processor Passthrough
"""

from __future__ import annotations
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsDetectionTransform:
    def __init__(self, augment: A.Compose) -> None:
        self.augment = augment

    def __call__(self, image, target):
        # 1. BULLETPROOF IMAGE SANITIZATION
        if not isinstance(image, np.ndarray):
            image_np = np.array(image)
        else:
            image_np = image.copy()
            
        # Scrub any possible NaNs from corrupted image decodes
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=255.0, neginf=0.0)
        if image_np.dtype != np.uint8:
            if image_np.max() <= 5.0: 
                image_np = np.interp(image_np, (image_np.min(), image_np.max()), (0, 255))
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            
        img_h, img_w = image_np.shape[:2]

        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()

        # 2. READ COCO ABSOLUTE FROM JSON AND CLIP
        safe_boxes = []
        safe_labels = []
        for box, label in zip(boxes, labels):
            x, y, w, h = box
            
            x = max(0.0, min(float(x), float(img_w - 1.0)))
            y = max(0.0, min(float(y), float(img_h - 1.0)))
            w = max(0.0, min(float(w), float(img_w - x)))
            h = max(0.0, min(float(h), float(img_h - y)))
            
            # Pre-augmentation safety check: discard sub-2-pixel slivers
            if w >= 2.0 and h >= 2.0:
                safe_boxes.append([x, y, w, h])
                safe_labels.append(label)

        # 3. RUN ALBUMENTATIONS (format="coco")
        transformed = self.augment(
            image=image_np,
            bboxes=safe_boxes,
            labels=safe_labels,
        )

        out_img = transformed["image"]
        if not isinstance(out_img, torch.Tensor):
            out_img = torch.from_numpy(out_img)
        if out_img.dtype == torch.uint8:
            out_img = out_img.float() / 255.0
        if out_img.ndim == 3 and out_img.shape[-1] == 3:
            out_img = out_img.permute(2, 0, 1)

        new_boxes = transformed["bboxes"]
        new_labels = transformed["labels"]

        # 4. STRICT POST-AUGMENTATION AREA CLAMPING (DEFORMABLE DETR FIX)
        pascal_boxes = []
        _, final_h, final_w = out_img.shape 
        
        for b in new_boxes:
            x, y, w_box, h_box = b
            
            # Deformable DETR GIoU Loss Explodes to NaN if Area == 0
            # We mathematically force the width and height to be at least 2 pixels
            w_box = max(float(w_box), 2.0)
            h_box = max(float(h_box), 2.0)
            
            # Convert to Pascal VOC [x1, y1, x2, y2]
            x1 = float(x)
            y1 = float(y)
            x2 = min(x1 + w_box, float(final_w))
            y2 = min(y1 + h_box, float(final_h))
            
            # If clipping to the image edge crushed the box, shift it inward safely
            if (x2 - x1) < 2.0: x1 = max(0.0, x2 - 2.0)
            if (y2 - y1) < 2.0: y1 = max(0.0, y2 - 2.0)
                
            pascal_boxes.append([x1, y1, x2, y2])

        if len(pascal_boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        else:
            target["boxes"] = torch.tensor(pascal_boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(new_labels, dtype=torch.int64)

        return out_img, target


def get_train_transform_detr_minimal():
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_area=4.0, min_visibility=0.1))
    return AlbumentationsDetectionTransform(train_tf)


def get_train_transform_detr_light():
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_area=4.0, min_visibility=0.1))
    return AlbumentationsDetectionTransform(train_tf)


def get_train_transform_detr_medium():
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-15, 15), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_area=4.0, min_visibility=0.1))
    return AlbumentationsDetectionTransform(train_tf)


def get_val_transform_detr():
    val_tf = A.Compose([ToTensorV2()], bbox_params=A.BboxParams(format="coco", label_fields=["labels"], min_visibility=0.0))
    return AlbumentationsDetectionTransform(val_tf)

def get_progressive_transform(epoch: int):
    if epoch < 20: return get_train_transform_detr_minimal()
    elif epoch < 50: return get_train_transform_detr_light()
    else: return get_train_transform_detr_medium()