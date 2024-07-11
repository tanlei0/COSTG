<a href=tmp><img src="https://img.shields.io/badge/arxiv-2312.07381-orange?logo=arxiv&logoColor=white"/></a>
<a href="https://huggingface.co/spaces/QinLei086/Curvilinear_Object_Generation_by_Text_and_Segmap"><img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>
<a href="https://huggingface.co/datasets/QinLei086/COSTG_v1"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue"></a>
# COSTG
The repository of **ECCV 2024** paper titled ***Enriching Information and Preserving Semantic Consistency in Expanding Curvilinear Object Segmentation Datasets***.

## COSTG Dataset
COSTG is a text-image generation dataset for curvilinear objects.
![Dataset](https://raw.githubusercontent.com/tanlei0/COSTG/main/figs/data_examples.jpg)

COSTG dataset can be download in <a href="https://huggingface.co/datasets/QinLei086/COSTG_v1"><img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-blue"></a>.

## Pipeline to generate new samples
In this paper, we introduce the Semantic Consistency Preserving ControlNet (SCP ControlNet), which adapts the SPADE module to maintain consistency between synthetic images and semantic maps.
![Pipeline](https://raw.githubusercontent.com/tanlei0/COSTG/main/figs/pipeline.jpg)

## Demo
A Huggingface demo <a href="https://huggingface.co/spaces/QinLei086/Curvilinear_Object_Generation_by_Text_and_Segmap"><img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>

## Training
under construction

## To Do
- [x] Release Gradio demo 
- [x] Release COSTG dataset
- [ ] Release model code and weights
- [ ] Release training and inference code

## Acknowledgements

Code for COSTG builds on [ControlNet](https://github.com/lllyasviel/ControlNet) and [SPADE](https://github.com/NVlabs/SPADE). Thanks to the authors for open-sourcing models. 

## Citation
under construction

## Contact
For any problem about this dataset or codes, please contact Dr. Qin Lei (qinlei@cqu.edu.cn, tanlei086@gmail.com)


