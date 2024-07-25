<a href="https://arxiv.org/abs/2407.08209"><img src="https://img.shields.io/badge/arxiv-2407.08209-orange?logo=arxiv&logoColor=white"/></a>
<a href="https://huggingface.co/spaces/QinLei086/Curvilinear_Object_Generation_by_Text_and_Segmap"><img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>
<a href="https://huggingface.co/datasets/QinLei086/COSTG_v1"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue"></a>
# COSTG
The repository of **ECCV 2024** paper titled [***Enriching Information and Preserving Semantic Consistency in Expanding Curvilinear Object Segmentation Datasets***](https://arxiv.org/abs/2407.08209).

## COSTG Dataset
COSTG is a text-image generation dataset for curvilinear objects.
![Dataset](https://raw.githubusercontent.com/tanlei0/COSTG/main/figs/data_examples.jpg)

COSTG dataset can be download in <a href="https://huggingface.co/datasets/QinLei086/COSTG_v1"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue"></a>.

## Pipeline to generate new samples
In this paper, we introduce the Semantic Consistency Preserving ControlNet (SCP ControlNet), which adapts the SPADE module to maintain consistency between synthetic images and semantic maps.
![Pipeline](https://raw.githubusercontent.com/tanlei0/COSTG/main/figs/pipeline.jpg)

## Demo and Model Weights
A Huggingface demo <a href="https://huggingface.co/spaces/QinLei086/Curvilinear_Object_Generation_by_Text_and_Segmap"><img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>

SD1.5 [UNet](https://huggingface.co/spaces/QinLei086/Curvilinear_Object_Generation_by_Text_and_Segmap/tree/main/pretrain_weights/unet)

[SCP_ControlNet](https://huggingface.co/spaces/QinLei086/Curvilinear_Object_Generation_by_Text_and_Segmap/tree/main/pretrain_weights/controlnet)
## Training

### Requirements
Before starting training, set up your environment by installing the necessary dependencies from the `requirements.txt` file in this repository. You can do this within an existing virtual environment using the following command:
```bash
pip install -r requirements.txt
```
This ensures all required libraries and frameworks are properly installed.

### Dataset setup
Ensure that the dataset format aligns with the specifications outlined in the [COSTG dataset on Hugging Face](https://huggingface.co/datasets/QinLei086/COSTG_v1). It is crucial that the data structure matches the expected format to facilitate proper training.

### Training Command
Training is conducted in two main stages:

1. **Fine-tuning a Standard SD1.5 Model:**
   Run the following command to fine-tune:
   ```bash
   python train_text_to_image.py --data_root "your data_root path"
   ```
   The trained weights will be saved to the directory specified by the `--output_dir` argument, which defaults to `sd-model-finetuned`.

2. **Training the SCP_ControlNet:**
   Upon completion of the SD1.5 model training, use its weights to train the SCP_ControlNet:
   ```bash
   python train_scp_controlnet.py --data_root "your data_root path" --unet_ckpt_path "your_root_path/sd-model-finetuned/checkpoint-xxx/unet"
   ```
   Weights are saved in the directory specified by the `--output_dir` argument, defaulting to `SCP-Control-Ouput`.

### Hyperparameter Configuration
To adjust parameters for fine-tuning SD1.5 and training SCP_ControlNet, refer to the configuration files located at `utils/args_parse_txt2img.py` and `utils/args_parse_curvilinear.py`.

### SCP ControlNet Training Trick ***(Important for training)***
When training SCP_ControlNet, integration of the SPADE module directly into ControlNet's Encoder Block may initially cause optimization direction conflicts due to SPADE's normalization method for adding semantic image information. To address this, we recommend the following steps:

- First, fine-tune a vanilla ControlNet model (refer to the Hugging Face tutorial on [training ControlNet](https://huggingface.co/docs/diffusers/training/controlnet)).
- Load the fine-tuned vanilla ControlNet weights and only randomly initialize the weights related to SPADE for further training.

You can use the following command with ***--controlnet_model_name_or_path*** to load the vanilla ControlNet weight when train SCP_ControlNet,after fine-tuning the vanilla ControlNet and obtaining the weight file:
```bash
python train_scp_controlnet.py --data_root "your data root path" --unet_ckpt_path "your root_path/sd-model-finetuned/checkpoint-xxx/unet" --controlnet_model_name_or_path "your_controlnet_weights/checkpoint-xxx/controlnet"
```

## Inference
Please refer the Huggingface demo <a href="https://huggingface.co/spaces/QinLei086/Curvilinear_Object_Generation_by_Text_and_Segmap"><img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>

## Acknowledgements

Code for COSTG builds on [ControlNet](https://github.com/lllyasviel/ControlNet) and [SPADE](https://github.com/NVlabs/SPADE). Thanks to the authors for open-sourcing models. 

## Citation
If you find our work or any of our materials useful, please cite our paper:
```
@misc{lei2024enrichinginformationpreservingsemantic,
      title={Enriching Information and Preserving Semantic Consistency in Expanding Curvilinear Object Segmentation Datasets}, 
      author={Qin Lei and Jiang Zhong and Qizhu Dai},
      year={2024},
      eprint={2407.08209},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.08209}, 
}
```

## Contact
For any problem about this dataset or codes, please contact Dr. Qin Lei (qinlei@cqu.edu.cn, tanlei086@gmail.com)


