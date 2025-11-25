import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# 检查是否有可用的GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载ResNet模型并去掉分类头
resnet_model = models.resnet50(pretrained=True)
resnet_model.fc = torch.nn.Identity()  # 去掉全连接层
resnet_model = resnet_model.to(device)
resnet_model.eval()  # 进入评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 裁剪图像函数
def crop_image(image, mask):
    # 将mask转换为numpy数组
    mask_np = np.array(mask)

    # 找到白色像素（假设255表示目标区域）
    white_pixels = np.where(mask_np == 255)

    # 如果没有白色像素，返回原图
    if len(white_pixels[0]) == 0:
        print("No white pixels found.")
        return image

    # 计算最小外接矩形的坐标
    x_min, x_max = np.min(white_pixels[1]), np.max(white_pixels[1])
    y_min, y_max = np.min(white_pixels[0]), np.max(white_pixels[0])

    # 设置扩展的边距（例如10像素）
    padding = 10

    # 确保扩大后的矩形仍在图像范围内
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.width)
    y_max = min(y_max + padding, image.height)

    # 裁剪扩展后的图像
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    return cropped_image

# 读取图像和mask
def load_image_and_mask(image_path, use_crop=False):
    image = Image.open(image_path).convert('RGB')
    if use_crop:
        mask_path = image_path.replace("images", "masks")  # 假设mask和图像在不同文件夹
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            image = crop_image(image, mask)
    return preprocess(image)

# 从文件夹中获取图像路径
def get_image_paths_from_folder(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# 提取图像特征
def extract_features(image_paths, use_crop=False):
    features = []
    for image_path in image_paths:
        image = load_image_and_mask(image_path, use_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = resnet_model(image)
        features.append(feature.squeeze(0))
    return torch.stack(features)

# 计算并保存ResNet特征的平均特征向量
def compute_and_save_resnet_prototype(folder_path, prototype_save_path, use_crop=False):
    image_paths = get_image_paths_from_folder(folder_path)
    if not image_paths:
        raise ValueError("文件夹中没有找到任何图像文件。")
    features = extract_features(image_paths, use_crop=use_crop)
    prototype = features.mean(dim=0)
    torch.save(prototype, prototype_save_path)
    print("Prototype saved at:", prototype_save_path)

# 示例文件夹路径
# folder_path = '/data1/wzb/hardsamply/contorl-lorav3/gen/new/withoutcrop'
folder_path = "/data1/wzb/dataset/polyp/polyp_dataset/traindata_size100/images"
# folder_path = "/data1/wzb/hardsamply/contorl-lorav3/gen/new/crop"
prototype_save_path = "/home/wzb/workspace/control-lora-v3/utils/Resnet/ave_polyp_resnet_prototype.pt"

# 计算并保存ResNet特征的平均特征向量，带裁剪
compute_and_save_resnet_prototype(folder_path, prototype_save_path, use_crop=True)
