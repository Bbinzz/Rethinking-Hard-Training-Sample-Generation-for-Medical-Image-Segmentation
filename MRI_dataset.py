import pandas as pd
import datasets
import os
import logging
import json
# 数据集路径设置
META_DATA_PATH = "/home/wzb/workspace/Unet_liver_seg-master/Unet_liver_seg-master/data/train/traindata.json"
IMAGE_DIR = "/home/wzb/workspace/Unet_liver_seg-master/Unet_liver_seg-master/data/train"
CONDITION_IMAGE_DIR = "/home/wzb/workspace/Unet_liver_seg-master/Unet_liver_seg-master/data/train"


# 定义数据集中有哪些特征，及其类型
_FEATURES = datasets.Features(
    {
        "source": datasets.Image(),
        "target": datasets.Image(),
        "prompt": datasets.Value("string"),
    },
)


# 定义数据集
class MRI(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [datasets.BuilderConfig(name="default", version=datasets.Version("0.0.2"))]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description="None",
            features=_FEATURES,
            supervised_keys=None,
            homepage="None",
            license="None",
            citation="None",
        )

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": META_DATA_PATH,
                    "images_dir": IMAGE_DIR,
                    "conditioning_images_dir": CONDITION_IMAGE_DIR,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        # metadata = pd.read_json(metadata_path, lines=True)[0]
        with open(metadata_path, 'rt') as f:
            metadata = json.load(f)

        for row in metadata:
            text = row["prompt"]

            image_path = row["target"]
            image_path = os.path.join(images_dir, image_path)

            # 打开文件错误时直接跳过
            try:
                image = open(image_path, "rb").read()
            except Exception as e:
                logging.error(e)
                continue

            conditioning_image_path = os.path.join(
                conditioning_images_dir, row["source"]
            )

            # 打开文件错误直接跳过
            try:
                conditioning_image = open(conditioning_image_path, "rb").read()
            except Exception as e:
                logging.error(e)
                continue

            yield row["target"], {
                "prompt": text,
                "target": {
                    "path": image_path,
                    "bytes": image,
                },
                "source": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }
