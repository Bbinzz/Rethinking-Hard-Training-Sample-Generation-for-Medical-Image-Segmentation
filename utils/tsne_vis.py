import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import cv2
import os

# 读取图片数据并转换为特征向量
def read_images_as_vectors(file_paths):
    vectors = []
    for file_path in file_paths:
        # 读取图片并调整大小为512x512
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        # 将图片数据拉平为一维向量
        vector = image.flatten()
        vectors.append(vector)
    return np.array(vectors)

# 随机选择x个样本
def random_sample(file_paths, num_samples=100):
    return np.random.choice(file_paths, num_samples, replace=False)

# 读取图片数据
org_data_path = "/data1/wzb/dataset/polyp/polyp_dataset/traindata_size100/images"
gen_data_path = "/data1/wzb/hardsamply/contorl-lorav3/gen/lora/images"
third_data_path = "/data1/wzb/hardsamply/contorl-lorav3/gen/sdloss+disloss_margin=50_a=0.0001/images"

org_file_paths = [os.path.join(org_data_path, file) for file in os.listdir(org_data_path)][0:100]
gen_file_paths = [os.path.join(gen_data_path, file) for file in os.listdir(gen_data_path)][0:1000]
third_file_paths = [os.path.join(third_data_path, file) for file in os.listdir(third_data_path)][0:1000]

org_vectors = read_images_as_vectors(org_file_paths)
gen_vectors = read_images_as_vectors(gen_file_paths)
third_vectors = read_images_as_vectors(third_file_paths)

# 合并数据
# all_vectors = np.vstack([org_vectors, gen_vectors, third_vectors])
# all_labels = np.array(["Real"] * len(org_vectors) + ["w/o disloss"] * len(gen_vectors) + ["w disloss"] * len(third_vectors))

#合并数据2
all_vectors_one = np.vstack([org_vectors, gen_vectors])
all_labels_one = np.array(["Real"] * len(org_vectors) + ["w/o disloss"] * len(gen_vectors))

all_vectors_2 = np.vstack([org_vectors, third_vectors])
all_labels_2 = np.array(["Real"] * len(org_vectors) + ["w disloss"] * len(third_vectors))

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results_one = tsne.fit_transform(all_vectors_one)
tsne_results_2 = tsne.fit_transform(all_vectors_2)

# 创建一个包含两个子图的图形
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# 第一个子图：绘制 "Real" 和 "w/o disloss Generated" 的 t-SNE 图
axes[0].set_title("Real vs w/o disloss", fontdict={'size': 25})
for label, c in zip(["Real", "w/o disloss"], ["red", "blue"]):
    indices = np.where(all_labels_one == label)
    axes[0].scatter(tsne_results_one[indices, 0], tsne_results_one[indices, 1], label=label, c=c)
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].legend(prop={'size': 15})

# 第二个子图：绘制 "Real" 和 "w disloss Generated" 的 t-SNE 图
axes[1].set_title("Real vs w disloss", fontdict={'size': 25})
for label, c in zip(["Real", "w disloss"], ["red", "green"]):
    indices = np.where(all_labels_2 == label)
    axes[1].scatter(tsne_results_2[indices, 0], tsne_results_2[indices, 1], label=label, c=c)
axes[1].tick_params(axis='both', which='major', labelsize=20)
axes[1].legend(prop={'size': 15})

# 保存图像
plt.savefig("pic/size1000.png")
plt.show()
