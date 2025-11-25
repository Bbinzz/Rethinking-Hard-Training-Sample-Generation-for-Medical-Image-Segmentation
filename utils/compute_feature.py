import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import numpy as np
import os
import re

# 设备选择
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 加载模型
local_model_dir = '/data1/wzb/pretrainmodel'
feature_extractor = ViTFeatureExtractor.from_pretrained(os.path.join(local_model_dir, 'vit-large-patch16-224'))
model = ViTModel.from_pretrained(os.path.join(local_model_dir, 'vit-large-patch16-224')).to(device)

# 固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

# 预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# 健壮裁剪函数
def crop_image(image, mask, padding=10, threshold=127):
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    
    # 以阈值做二值化
    binary_mask = (mask_np > threshold).astype(np.uint8)
    white_pixels = np.where(binary_mask == 1)

    if len(white_pixels[0]) == 0:
        print("No white pixels found, returning original image")
        return None

    x_min, x_max = np.min(white_pixels[1]), np.max(white_pixels[1])
    y_min, y_max = np.min(white_pixels[0]), np.max(white_pixels[0])

    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.width)
    y_max = min(y_max + padding, image.height)

    # 裁剪合法性校验
    if x_max <= x_min or y_max <= y_min:
        print(f"Invalid crop region: ({x_min},{y_min},{x_max},{y_max}). Return original image.")
        return None

    return image.crop((x_min, y_min, x_max, y_max))

# 加载图像
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return preprocess(image)

# 加载mask
def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    return mask

# 特征提取
def extract_features(image_paths, mask_paths, use_mask=False):
    features = []
    for img_path, msk_path in zip(image_paths, mask_paths):
        image = Image.open(img_path).convert('RGB')
        mask = load_mask(msk_path)

        if use_mask:
            image = crop_image(image, mask)
            if not image:
                continue
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
        features.append(outputs.last_hidden_state[:, 0, :].squeeze(0))
    return torch.stack(features)

# 计算均值
def compute_average_features(image_paths, mask_paths):
    features = extract_features(image_paths, mask_paths, use_mask=True)
    return features.mean(dim=0)

# 从文件夹取图像路径
def get_image_paths_from_folder(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)  # 排序，确保顺序一致

# 匹配image和mask的顺序
def match_image_mask(image_folder, mask_folder):
    image_files = get_image_paths_from_folder(image_folder)
    mask_files = get_image_paths_from_folder(mask_folder)

    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_files}
    mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_files}

    # for p in mask_files:
    #     basename = os.path.splitext(os.path.basename(p))[0]
    #     if basename.endswith("_Segmentation"):
    #         image_key = basename.replace("_Segmentation", "")
    #         mask_dict[image_key] = p

    common_keys = sorted(list(set(image_dict.keys()) & set(mask_dict.keys())))

    image_paths = [image_dict[k] for k in common_keys]
    mask_paths = [mask_dict[k] for k in common_keys]

    print(f"共匹配到 {len(common_keys)} 对图像和mask")
    return image_paths, mask_paths

# 主流程
image_folder = '/data1/wzb/dataset/Tumar_CT/train/images'
mask_folder = '/data1/wzb/dataset/Tumar_CT/train/masks'
image_paths, mask_paths = match_image_mask(image_folder, mask_folder)

average_features = compute_average_features(image_paths, mask_paths)
print("Average Features:", average_features)
torch.save(average_features, "/home/wzb/workspace/control-lora-v3/prototype_result/tumar/tumar.pt")
