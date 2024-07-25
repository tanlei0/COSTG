from transformers import PretrainedConfig
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import os
import torch
from diffusers import (
    UniPCMultistepScheduler,
)
import numpy as np
from model.ControlnetPipeline import StableDiffusionControlNetPipeline_SPADE
from PIL import Image, ImageDraw, ImageFont
import random
import textwrap

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
        

def tokenize_captions(tokenizer, captions):

    inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
    return inputs.input_ids.squeeze()


def get_transforms(resolution=512):
    transforms = A.Compose([
        A.Resize(resolution, resolution),
        A.CenterCrop(resolution, resolution),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    return transforms


class BaseImgControlnet(Dataset):
    def __init__(self, data_root, json_file, tokenizer, resolution = 512):
        self.data_root = data_root
        self.data = self._load_data(os.path.join(data_root,json_file))
        self.transform = get_transforms(resolution)
        self.tokenizer = tokenizer

        self.data = [entry for entry in self.data if "images" in entry["image"]]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image_path = os.path.join(self.data_root, item["image"])
        conditioning_image_path = os.path.join(self.data_root, item["image"].replace("images","annotations"))
        conditioning_image_path = img_path2anno_path(conditioning_image_path, image_path)

        image = self._load_image(image_path)
        conditioning_image = self._load_image(conditioning_image_path)

        augmented = self.transform(image=np.array(image))
        image = augmented["image"]
        conditioning_image = np.array(conditioning_image) / 255
        conditioning_image = torch.from_numpy(conditioning_image).permute(2,0,1)
        if image.shape != conditioning_image.shape:
            raise "shape error"
        text = item["caption"]
        processed_text = tokenize_captions(self.tokenizer, [text])

        return {"pixel_values": image, "conditioning_pixel_values": conditioning_image, "input_ids": processed_text}

    def _load_data(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image
    
def img_path2anno_path(conditioning_image_path,image_path):
    if "crack" in image_path:
        conditioning_image_path = conditioning_image_path.replace("jpg","png")
    elif any(substring in image_path for substring in ["CHASEDB1", "DRIVE", "CHUAC", "DCA1"]):
        conditioning_image_path = conditioning_image_path.replace("img","gt")
    return conditioning_image_path

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


def log_validation(vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, logger):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline_SPADE.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    for dataset in ["angiography","crack","retina"]:
        for val_step in range(args.val_images_limits):
            images = []
            img_entry, _= select_random_data(os.path.join(args.data_root,dataset,dataset+".json"))
            validation_image_file= os.path.join(args.data_root,dataset,img_entry["image"].replace("images", "annotations"))
            validation_image_file = img_path2anno_path(validation_image_file, validation_image_file)
            validation_image = Image.open(validation_image_file).convert("RGB")
            validation_prompt = img_entry["caption"]

            images = []
            images.append(validation_image)
            for _ in range(args.num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(
                        validation_prompt, validation_image, num_inference_steps=20, generator=generator
                    ).images[0]

                images.append(image)


            filename = "{}_{}.png".format(dataset,val_step)
            path = os.path.join(accelerator.logging_dir,"step_{}".format(step), filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            save_single_img(images, path, dataset, val_step, validation_prompt)
            merge_images_with_prompt(images, validation_prompt, path)

   
def save_single_img(image_list, path, dataset,val_step,validation_prompt):
    single_path = os.path.join(os.path.split(path)[0], "single")
    os.makedirs(single_path, exist_ok=True)

    for i, img in enumerate(image_list):
        name = "{}_{}_{}.png".format(dataset, val_step, i)
        img.save(os.path.join(single_path,name))
    name_txt ="{}_{}.txt".format(dataset, val_step)
    with open(os.path.join(single_path, name_txt), "w",encoding='utf-8') as file:
        file.write(validation_prompt)
def merge_images_with_prompt(image_list, prompt, output_path):
    # 创建底图
    width = sum(img.width for img in image_list) + 200
    max_height = max(img.height for img in image_list)
    background = Image.new('RGB', (width, max_height), 'white')

    # 在底图上粘贴四张图
    x_offset = 0
    for img in image_list:
        background.paste(img, (x_offset, 0))
        x_offset += img.width

     # 在右边添加包含换行的字符串 prompt 的图像
    prompt_image = Image.new('RGB', (200, max_height), 'white')
    draw = ImageDraw.Draw(prompt_image)
    font = ImageFont.load_default()

    # 文本换行处理
    wrap_text = textwrap.fill(prompt, width=30)  # 假设每行大约30个字符宽，根据需要调整
    draw.text((10, 10), wrap_text, font=font, fill='black')  # 文本从左上角开始

    # 将字符串图像粘贴到底图上
    background.paste(prompt_image, (width - 200, 0))

    # 保存合并后的图像
    background.save(output_path)


def select_random_data(json_file_path):

    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
    
    images_data = [item for item in json_data if "images" in item["image"]]
    annotations_data = [item for item in json_data if "annotations" in item["image"]]
    
    selected_image_data = random.choice(images_data)
    selected_annotation_data = random.choice(annotations_data)
    
    return selected_image_data, selected_annotation_data
