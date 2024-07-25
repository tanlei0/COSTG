from diffusers import StableDiffusionPipeline, DDPMScheduler
import torch
import numpy as np
import os, json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import random
import textwrap

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch, logger):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    
    for dataset in ["angiography","crack","retina"]:
        images_img = []
        images_anno = []
        img_entry, anno_entry= select_random_data(os.path.join(args.data_root,dataset,dataset+".json"))
        for _ in range(args.val_num):
            with torch.autocast("cuda"):
                image_img = pipeline(img_entry["caption"], num_inference_steps=20, generator=generator).images[0]
                image_anno = pipeline(anno_entry["caption"], num_inference_steps=20, generator=generator).images[0]

            images_img.append(image_img)
            images_anno.append(image_anno)

        for t in ["img", "anno"]:
            filename = "{}_{}_epoch_{}.png".format(dataset,t,epoch)
            path = os.path.join(accelerator.logging_dir,"val_images",filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            merge_images_with_prompt(images_img if t== "img" else images_anno, img_entry["caption"] if t== "img" else anno_entry["caption"], path)

    del pipeline
    torch.cuda.empty_cache()

def merge_images_with_prompt(image_list, prompt, output_path):
    # 创建底图
    width = sum(img.width for img in image_list) + 200  # 为prompt预留200像素宽度
    max_height = max(img.height for img in image_list)
    background = Image.new('RGB', (width, max_height), 'white')

    # 在底图上粘贴图像
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


class BaseDataset(Dataset):
    def __init__(self, data_root, json_file, tokenizer, resolution = 512):
        self.data_root = data_root
        self.data = self._load_data(os.path.join(data_root,json_file))
        self.transform = get_transforms(resolution)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image_path = os.path.join(self.data_root, item["image"])
        
        image = self._load_image(image_path)
        if "annotations" in item["image"]:
            image = np.array(image)
            if len(np.unique(image)) == 2:
                image = torch.from_numpy(image / 255)
                image = image.permute(2,1,0)
            else:
                raise "len of GT is not 2"
        else:
            augmented = self.transform(image=np.array(image))
            image = augmented["image"]
        text = item["caption"]
        processed_text = tokenize_captions(self.tokenizer, [text])

        return {"pixel_values": image, "input_ids": processed_text}

    def _load_data(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data

    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

class BaseSegDataset(BaseDataset):
    def __init__(self, data_root, json_file, tokenizer, resolution=512):
        super().__init__(data_root, json_file, tokenizer, resolution)
        filtered_data = [entry for entry in self.data if "annotations" in entry["images"]]
        self.data = filtered_data
        


