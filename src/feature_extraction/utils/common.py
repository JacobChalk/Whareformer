import numpy as np
import torch
import torch.nn.functional as F
import cv2

def create_object_crop(
    frame, 
    bbox, 
    input_mask,
    apply_mask=True,
    scale_frame_to_mask=True,
    pad_to_square=True
):
    frame_h, frame_w = frame.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    mask_h, mask_w = input_mask.shape[:2]  
    scaling_required = (frame_w != mask_w) or (frame_h != mask_h)

    if scaling_required:
        if scale_frame_to_mask:
            frame = cv2.resize(frame, (mask_w, mask_h), interpolation=cv2.INTER_CUBIC)
            w, h = mask_w, mask_h
        else:
            frame = frame.copy()
            x_scale = frame_w / mask_w
            y_scale = frame_h / mask_h

            x_min = x_min * x_scale
            x_max = x_max * x_scale
            y_min = y_min * y_scale
            y_max = y_max * y_scale

            w, h = frame_w, frame_h
    else:
        frame = frame.copy()
        w, h = frame_w, frame_h

    if apply_mask:
        mask = input_mask.astype(np.uint8)

        if scaling_required and not scale_frame_to_mask:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.uint8)
        frame = frame * mask[..., None]

    x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
    x_max, y_max = min(w, int(x_max)), min(h, int(y_max))

    if x_max <= x_min or y_max <= y_min:
        x_min, y_min = max(0, x_min - 1), max(0, y_min - 1)
        x_max = min(w, x_min + 1)
        y_max = min(h, y_min + 1)
        
    obj_crop = frame[y_min:y_max, x_min:x_max]

    crop_tensor = torch.from_numpy(obj_crop).permute(2, 0, 1).float() / 255.0

    if pad_to_square:
        _, h, w = crop_tensor.shape
        max_dim = max(h, w)
        
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        
        crop_tensor = F.pad(crop_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    return crop_tensor