<center>
<h1>Rethinking-Hard-Training-Sample-Generation-for-Medical-Image-Segmentation
</h1>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320325011963">[article]</a>
<a href="https://pan.baidu.com/s/15Divpt6B3SOnzkf_L_PkpQ?pwd=1111">[checkpoint]</a>
<a herf="https://www.modelscope.cn/studios/bbinzz/polyp-generation">[Demo]</a>
</center>

***Note:*** Extraction code is 1111

## Introduction

This repository contains the official implementation of our paper, which addresses the challenge of generating effective synthetic training data for segmentation tasks in data-scarce domains such as medical image analysis.

While prior methods dynamically adjust synthetic sample difficulty using the downstream model to prevent performance saturation, they often overlook the interoperability of generated samples. Samples tailored to one model may lack general challenge due to differing feature focuses across architectures.

To overcome this, we propose a novel generation strategy that uses the discrepancy between backbone-extracted features and real image prototypes to create broadly challenging samples.


## Pretrained Models

[control-lora-v3 sd canny](https://huggingface.co/HighCWu/sd-control-lora-v3-canny)

[control-lora-v3 sdxl canny](https://huggingface.co/HighCWu/sdxl-control-lora-v3-canny)

[control-lora-v3 pretrained models collection](https://huggingface.co/HighCWu/control-lora-v3)

Try `exps/test_<TASK>.py` to test different type conditions.

Try `exps/test_multi_load.py` to load multi lora and switch between them.

Try `exps/test_multi_inputs.py` to use multi lora at the same time.

I used a high learning rate and a short number of steps for training, and the dataset was also generated, so the generation results may not be very good. It is recommended that researchers use real data, lower learning and longer training steps to train to achieve better generation results.


## Prepare Environment

```sh
pip install -r environment.yml
```

## Prepare Dataset

downloading testing dataset and move it into ./data/TestDataset/, which can be found in this Google Drive Link (327.2MB). It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples), ETIS-LaribPolypDB (196 test samples), Kvasir (100 test samples).

downloading training dataset and move it into ./data/TrainDataset/, which can be found in this Google Drive Link (399.5MB). It contains two sub-datasets: Kvasir-SEG (900 train samples) and CVC-ClinicDB (550 train samples).

## Training

It's very easy to train control-lora-v3 stable diffusion or stable diffusion xl with resolution 512 images on a gpu with 16GB vram.

I put my detail training code in `exps/train_*.py`. You can refer to my code to configure the hyperparameters used for training, but the training data is local to my computer, so you need to modify `dataset_name` before you can use it normally.

For Stable Diffusion, use:

```sh
accelerate launch train.py \
 --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
 --output_dir=<YOUR_SAVE_DIR> \
 --tracker_project_name=<WANDB_PROJECT_NAME> \
 --dataset_name=<HF_DATASET> \
 --proportion_empty_prompts=0.1 \
 --conditioning_image_column=guide \
 --image_column=image --caption_column=text \
 --rank=16 \
 --lora_adapter_name=<YOUR_TASK_LORA_NAME> \
 --init_lora_weights=gaussian \
 --loraplus_lr_ratio=1.0 \
 --half_or_full_lora=half_skip_attn \
 --extra_lora_rank_modules conv_in \
 --extra_lora_ranks 64 \
 --resolution=512 \
 --learning_rate=0.0001 \
 --seed=42 \
 --validation_image <PATH_1> <PATH_2> <PATH_N> \
 --validation_prompt <PROMPT_1> <PROMPT_2> <PROMPT_N> \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --mixed_precision=fp16 \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --max_train_steps=50000 \
 --resume_from_checkpoint=latest \
 --report_to wandb \
 --push_to_hub
```

For Stable Diffusion XL, use: 

(***Note***: It will encounter the NaN problem when validating the generation results during the training process, but it does not affect the final training results. To be repaired.)

```sh
accelerate launch train_sdxl.py \
 --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
 --output_dir=<YOUR_SAVE_DIR> \
 --tracker_project_name=<WANDB_PROJECT_NAME> \
 --dataset_name=<HF_DATASET> \
 --proportion_empty_prompts=0.1 \
 --conditioning_image_column=guide \
 --image_column=image --caption_column=text \
 --rank=16 \
 --lora_adapter_name=<YOUR_TASK_LORA_NAME> \
 --init_lora_weights=gaussian \
 --loraplus_lr_ratio=1.0 \
 --half_or_full_lora=half_skip_attn \
 --extra_lora_rank_modules conv_in \
 --extra_lora_ranks 64 \
 --resolution=512 \
 --learning_rate=0.0001 \
 --seed=42 \
 --validation_image <PATH_1> <PATH_2> <PATH_N> \
 --validation_prompt <PROMPT_1> <PROMPT_2> <PROMPT_N> \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --mixed_precision=bf16 \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --max_train_steps=50000 \
 --resume_from_checkpoint=latest \
 --report_to wandb \
 --push_to_hub
```


You can init lora weights with the powerful [PiSSA](https://github.com/GraphPKU/PiSSA) by:
`
--init_lora_weights=pissa
`
or faster init with pissa_niter_n
`
--init_lora_weights=pissa_niter_2
`


You can custom the dataset training script by replace 
`
--dataset_name=<HF_DATASET>
`
with a script that includes a custom `torch.utils.data.Dataset` class:
`
--dataset_script_path=<YOUR_SCRIPT_PATH>
`.
You can refer to [`exps/sd1_5_tile_pair_data.py`](exps/sd1_5_tile_pair_data.py).


## Inference

For stable diffusion, use:

```python
# !pip install opencv-python transformers accelerate
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline
import numpy as np
import torch

import cv2
from PIL import Image

# download an image
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# load stable diffusion v1-5 and control-lora-v3 
unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16
)
unet = unet.add_extra_conditions(["canny"])
pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sd-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
# pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sd-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
pipe.load_lora_weights("HighCWu/sd-control-lora-v3-canny")

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
image = pipe(
    "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
).images[0]
image.show()
```

For stable diffusion xl, use:

```python
# !pip install opencv-python transformers accelerate
from diffusers import AutoencoderKL
from diffusers.utils import load_image
from model import UNet2DConditionModelEx
from pipeline_sdxl import StableDiffusionXLControlLoraV3Pipeline
import numpy as np
import torch

import cv2
from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

# download an image
image = load_image(
    "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

# initialize the models and pipeline
unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16
)
unet = unet.add_extra_conditions(["canny"])
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlLoraV3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, vae=vae, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sdxl-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
# pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sdxl-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
pipe.load_lora_weights("HighCWu/sdxl-control-lora-v3-canny")
pipe.enable_model_cpu_offload()

# get canny image
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# generate image
image = pipe(
    prompt, image=canny_image
).images[0]
image.show()
```
}

## Cite
```
@article{WAN2026112533,
title = {Rethinking hard training sample generation for medical image segmentation},
journal = {Pattern Recognition},
volume = {172},
pages = {112533},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112533},
author = {Zhibin Wan and Zhiqiang Gao and Mingjie Sun and Yang Yang and Cao Min and Hongliang He and Guohong Fu},
}
```









