import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)

batchsize = 2
task = "segmentation"
dataset_name = "isic"
proportion_empty_prompts = 0.1
rank = 128
learning_rate = 1e-4   # 默认为1e-4
half_or_full_lora = "half_skip_attn"
init_lora_weights = "gaussian"
extra_lora_rank_modules = ["conv_in"]
extra_lora_ranks = [128]
extra_suffix = "-".join([f"""{n}-rank{r * 2 if "pissa" in init_lora_weights else r}""" for n, r in zip(extra_lora_rank_modules, extra_lora_ranks)])
lora_adapter_name = f"""sd-control-lora-v3-{task}"""
resume_checkpoint="/data1/wzb/hardsamply/contorl-lorav3/checkpoint/isic/lora/checkpoint-5000"

use_disloss = "use"  # "no" or "use"
# repo_id = f"""sd-control-lora-v3-{task}-{half_or_full_lora}-rank{rank * 2 if "pissa" in init_lora_weights else rank}-{extra_suffix}"""

repo_id = f"""our"""

cmd = f"""\
CUDA_VISIBLE_DEVICES=1,2,3,4 \
accelerate launch --main_process_port=8888 train.py \
 --pretrained_model_name_or_path=/data1/wzb/pretrainmodel/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9\
 --isuse_disloss={use_disloss}\
 --output_dir="/data1/wzb/hardsamply/contorl-lorav3/checkpoint/isic/{repo_id}" \
 --tracker_project_name="{repo_id}" \
 --dataset_name={dataset_name} \
 --train_batch_size={batchsize} \
 --proportion_empty_prompts={proportion_empty_prompts} \
 --conditioning_image_column=source \
 --image_column=target \
 --caption_column=prompt \
 --rank={rank} \
 --lora_adapter_name={lora_adapter_name} \
 --init_lora_weights={init_lora_weights} \
 --half_or_full_lora={half_or_full_lora} \
 --extra_lora_rank_modules {" ".join(extra_lora_rank_modules)} \
 --extra_lora_ranks {" ".join([str(r) for r in extra_lora_ranks])} \
 --resolution=512 \
 --learning_rate={learning_rate} \
 --seed=42 \
 --mixed_precision=no \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=1000 \
 --validation_steps=10000 \
 --max_train_steps=8000 \
 --resume_from_checkpoint=latest\
 --report_to tensorboard \


"""
#   --resume_from_checkpoint={resume_checkpoint}\
print(cmd)
os.system(cmd)
