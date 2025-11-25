import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)

import time
from utils.gendataset import polypgenDataset
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from diffusers.utils import load_image
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from PIL import Image

from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline
import json
torch.manual_seed(42)
device = "cuda:5"

model_id = "/data1/wzb/pretrainmodel/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9" # "ckpt/anything-v3.0" # "runwayml/stable-diffusion-v1-5" # 
vae_model_path = "/data1/wzb/pretrainmodel/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9" # "ckpt/anything-v3.0" # "runwayml/stable-diffusion-v1-5" # 
vae_subfolder = "vae" # "vae" # "vae" # 

vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder=vae_subfolder, torch_dtype=torch.float16)

unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
unet = unet.add_extra_conditions("sd-control-lora-v3-segmentation")

pipe: StableDiffusionControlLoraV3Pipeline = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, safety_checker=None, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("/data1/wzb/hardsamply/feedback/control_lora_ck/segformer_a_0.01/checkpoint-7000")
pipe.load_lora_weights("/data1/wzb/hardsamply/contorl-lorav3/checkpoint/ablation_10-9", subfolder="checkpoint-5000")
pipe = pipe.to(device=device, dtype=torch.float16)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# result_root = "/data1/wzb/hardsamply/contorl-lorav3/isic/our"
result_root = "/data1/wzb/hardsamply/contorl-lorav3/gen/ablation_10-9"
# result_root = "/home/wzb/workspace/control-lora-v3/test/tumar/our/5000"
img_root = os.path.join(result_root,"images")
mask_root = os.path.join(result_root,"masks")
os.makedirs(img_root, exist_ok=True)
os.makedirs(mask_root,exist_ok=True)
batch_size = 8
def save_image(imgs,gen_index,root):
    for idx,img in enumerate(imgs):
        path = os.path.join(root,"{}--{}.png".format(gen_index,idx)) 
        img.save(path)
        
def save_mask(masks,gen_index,root):
    for idx,mask in enumerate(masks):
        mask = mask.numpy()
        path = os.path.join(root,"{}--{}".format(gen_index,idx))
        Image.fromarray(mask).convert('1').save(path)


# dataroot = "/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/"
# datajson = "/data1/wzb/dataset/polyp/polyp_dataset/TrainDataset/data.json"
# dataroot = "/data1/wzb/dataset/ISIB2016/train"
# datajson = "/data1/wzb/dataset/ISIB2016/train/traindata.json"
dataroot = '/home/wzb/workspace/Unet_liver_seg-master/Unet_liver_seg-master/data/train'
datajson = '/home/wzb/workspace/Unet_liver_seg-master/Unet_liver_seg-master/data/train/traindata.json'
# dataset = polypgenDataset(datajson,dataroot)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

with open(datajson, 'rt') as f:
    data = json.load(f)
for idx, batch in enumerate(data):
    # if idx == 10:
    #     break
    prompt = batch["prompt"]
    conditionmask_path= batch["source"]
    #### rotated
    # conditionmask = load_image(os.path.join(dataroot,conditionmask_path)).resize((512,512),Image.Resampling.BILINEAR).rotate(90, expand=True)
    conditionmask = load_image(os.path.join(dataroot,conditionmask_path)).resize((512,512),Image.Resampling.BILINEAR)
    # start_time = time.time()
    output = pipe(
        prompt,
        conditionmask,
        num_inference_steps=20,
    )
    # end_time = time.time()
    # print("用时：",end_time-start_time)
    # print(output)
    save_image(output.images,idx,img_root)

    conditionmask.convert("1")
    conditionmask.save(os.path.join(mask_root,"{}--{}.png".format(idx,0)))
    # save_mask(batch["mask"],idx,mask_root)
    


# images = [(path, load_image(path)) for path in ["./imgs/segmentation1.jpg", "./imgs/segmentation2.jpg", "./imgs/segmentation3.jpg", "./imgs/segmentation4.jpg"]]
# prompts = [
#     "realistic painting of a tardigrade kaiju, with 6 legs in a desert storm, by james gurney, slime, big globule eye, godzilla, vintage, concept art, oil painting, tonalism, crispy",
#     "portrait of a beautiful cute strong brave realistic! female gnome engineer, textured undercut black hair, d & d, micro detail, intricate, elegant, highly detailed, centered, rule of thirds, artstation, sharp focus, illustration, artgerm, tomasz alen kopera, donato giancola, wlop",
#     "beautiful digital painting of a stylish asian female forest with high detail, 8 k, stunning detail, works by artgerm, greg rutkowski and alphonse mucha, unreal engine 5, 4 k uhd",
#     "duotone noir scifi concept dynamic illustration of 3 d mesh of robotic cats inside box floating zero gravity glowing 3 d mesh portals futuristic, glowing eyes, octane render, surreal atmosphere, volumetric lighting. accidental renaissance. by sachin teng and sergey kolesov and ruan jia and heng z. graffiti art, scifi, fantasy, hyper detailed. trending on artstation"
# ]

# os.makedirs("./out/grids", exist_ok=True)
# for (path, image), prompt in zip(images, prompts):
#     prompt = "best quality, extremely detailed, " + prompt
#     generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(3)]

#     output = pipe(
#         [prompt]*3,
#         [image]*3,
#         negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 3,
#         num_inference_steps=20,
#         generator=generator,
#     )

    # grid = image_grid([image] + output.images, 2, 2)
    # grid.save("./out/grids/%s.png" % os.path.basename(path).split(".")[0])

