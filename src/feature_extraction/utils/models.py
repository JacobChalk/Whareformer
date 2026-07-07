import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
import torch.nn.functional as F
import numpy as np
import trimesh
import pyrender

from transformers import AutoModel, AutoModelForDepthEstimation, AutoImageProcessor, Sam2Model, Sam2Processor
from torchvision import transforms


# --- 2D Feature Model ---
def get_dino_style_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# --- 2D Model Forward Pass (Helpers) ---
    
def forward_dino_model_style(model, object_crops, strategy='cls'):
    features = model.forward_features(object_crops)
    if strategy == 'cls':
        return features['x_norm_clstoken']

    return features['x_norm_patchtokens'].mean(dim=1)

# --- 2D Model Factory ---

def load_2d_model(model_choice, device):
    print(f"Loading 2D model: {model_choice}")
    
    if model_choice == 'dinov2_giant':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        transform_2d = get_dino_style_transform()
        forward_fn = forward_dino_model_style

    else:
        raise ValueError(f"Unknown 2D model choice: {model_choice}")
    
    model = model.to(device).eval()
    print(f"Model {model_choice} loaded.")
    
    return model, transform_2d, forward_fn

# --- Instance Segmentation Model (SAM) ---

def forward_sam2_style(model, processor, images, boxes):
    final_masks = []
    
    for img, img_boxes in zip(images, boxes):
        inputs = processor(
            images=img,
            input_boxes=[img_boxes], 
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes=inputs["original_sizes"].tolist()
        )[0]
        final_masks.append(masks)

    return final_masks

def load_instance_segmentation_model(model_choice, device):
    if model_choice == "sam2":
        model = Sam2Model.from_pretrained("facebook/sam2-hiera-large")
        processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-large")
        forward_fn = forward_sam2_style
    else:
        raise ValueError(f"Unknown instance segmentation model choice: {model_choice}")
    
    model = model.to(device).eval()
    print(f"Model {model_choice} loaded.")
    
    return model, processor, forward_fn

# --- 3D Depth Model (DepthAnything) ---

def forward_depthanything_style(frames, model, processor, device, frame_size):
    inputs = processor(
        images=[frame for frame in frames], 
        return_tensors="pt", 
        do_rescale=False 
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    depth_maps = outputs.predicted_depth.unsqueeze(1)
    depth_maps = F.interpolate(
        depth_maps, 
        size=(frame_size[1], frame_size[0]), 
        mode='bicubic', 
        align_corners=False
    )

    # Clamp depth to 0.
    depth_maps = torch.relu(depth_maps)
    
    return depth_maps.squeeze(1)

def load_depth_model(device):
    print("Loading DepthAnything model and processor...")
    checkpoint = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
    model = model.to(device).eval()
    
    print("DepthAnything model loaded.")
    return model, processor, forward_depthanything_style

# --- 3D Scene Model (Pyrender) ---

class PyrenderScene:
    def __init__(self, mesh_path, camera_params, scale_factor=1.0):
        print("Loading 3D mesh and Pyrender scene...")
        self.width, self.height, fx, fy, cx, cy = camera_params
        
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_scale(scale_factor)
        
        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=cx, cy=cy,
            znear=0.01, zfar=10.0
        )
        
        self.scene = pyrender.Scene()
        pmesh = pyrender.Mesh.from_trimesh(mesh)
        self.scene.add(pmesh)
        self.camera_node = self.scene.add(camera, pose=np.eye(4))

        self.flip_yz = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ], dtype=np.float32)

        self.renderer = pyrender.OffscreenRenderer(self.width, self.height)
        print("Pyrender scene initialised.")

    def get_scene_depth(self, camera_pose_c2w):
        c2w_pose_opengl = camera_pose_c2w @ self.flip_yz
        
        self.scene.set_pose(self.camera_node, pose=c2w_pose_opengl)
        return self.renderer.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)