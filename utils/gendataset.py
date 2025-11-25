import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import random
from utils.rotate_mask import rotate_mask
from diffusers.utils import load_image
# dataloader更改
class polypgenDataset(Dataset):
    def __init__(self,datajson,root,train_num=-1,mask_rotated = False,angle=0):
        with open(datajson, 'rt') as f:
            self.data = json.load(f)
            if train_num > 0:
                self.data = sorted(self.data,key=lambda x: x['source'])[0:train_num]
        self.root = root
        self.mask_rotated = mask_rotated
        self.angle = angle
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        source = load_image(self.root + source_filename)
        target = cv2.imread(self.root + target_filename)

        maskpath = os.path.join(self.root,source_filename)
        if self.mask_rotated:
                #随机旋转mask
                mask = rotate_mask(maskpath,self.angle)
        else:
                mask = cv2.imread(maskpath,-1)
                mask = cv2.resize(mask,(512,512))
        
        # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # # print(source)
        # source = cv2.resize(source,(512,512))
        # target = cv2.resize(target,(512,512))
        # # Normalize source images to [0, 1].
        # source = source.astype(np.float32) / 255.0

        # # Normalize target images to [-1, 1].
        # target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(txt=prompt, condition=source,mask = mask)