import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
import os

# 检查是否有可用的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
 
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(42)
# 加载ViT模型和特征提取器
local_model_dir = '/data1/wzb/pretrainmodel'
feature_extractor = ViTImageProcessor.from_pretrained(os.path.join(local_model_dir, 'vit-large-patch16-224'))
model = ViTModel.from_pretrained(os.path.join(local_model_dir, 'vit-large-patch16-224')).to(device)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# 裁剪图像的函数
def crop_image(image, mask):
    mask_np = np.array(mask)
    white_pixels = np.where(mask_np == 255)
    if len(white_pixels[0]) == 0:
        print("No white pixels found.")
        return image
    x_min, x_max = np.min(white_pixels[1]), np.max(white_pixels[1])
    y_min, y_max = np.min(white_pixels[0]), np.max(white_pixels[0])
    padding = 10
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.width)
    y_max = min(y_max + padding, image.height)
    return image.crop((x_min, y_min, x_max, y_max))

# 读取和处理图像和掩码
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return preprocess(image)

def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    return mask

# 提取图像特征
def extract_features(image_paths, use_ask=False):
    features = []
    if use_ask:
        for image_path in image_paths:
            mask_path = image_path.replace("images", "masks")
            mask = load_mask(mask_path)
            image = Image.open(image_path).convert('RGB')
            cropped_image = crop_image(image, mask)
            cropped_image = preprocess(cropped_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(cropped_image)
            features.append(outputs.last_hidden_state[:, 0, :].squeeze(0))
        return torch.stack(features)
    else:
        for image_path in image_paths:
            image = load_image(image_path).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image)
            features.append(outputs.last_hidden_state[:, 0, :].squeeze(0))
        return torch.stack(features)

# 计算特征到已有均值特征（prototype）的距离均值和方差
def compute_distances_to_prototype(image_paths, prototype_path):
    prototype = torch.load(prototype_path).to(device)  # 加载已存储的prototype
    features = extract_features(image_paths, use_ask=True)
    distances = torch.norm(features - prototype, dim=1)
    avg_distance = distances.mean().item()
    variance = distances.var().item()
    return avg_distance, variance

# 从文件夹中获取图像路径
def get_image_paths_from_folder(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 计算图像特征到已有prototype的距离均值和方差
# folder_path = '/data1/wzb/hardsamply/contorl-lorav3/gen/new/withoutcrop/images'
# folder_path = "/data1/wzb/hardsamply/contorl-lorav3/gen/lora/images"
folder_path = "/data1/wzb/hardsamply/contorl-lorav3/gen/new/crop/images"
prototype_path = "/home/wzb/workspace/control-lora-v3/new_proto/ave_polyp_vit.pt"  # prototype文件路径
image_paths = get_image_paths_from_folder(folder_path)

# 输出平均距离和方差
avg_distance, var_distance = compute_distances_to_prototype(image_paths, prototype_path)
print(folder_path)
print("Average Distance to Prototype:", avg_distance)
print("Variance of Distances:", var_distance)
