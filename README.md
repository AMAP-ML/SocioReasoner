# SocioReasoner
[![arXiv](https://img.shields.io/badge/arXiv-2601.10477-b31b1b.svg)](https://arxiv.org/abs/2601.10477)

Official implementation of Urban Socio-Semantic Segmentation with Vision-Language Reasoning.

<img src="./assets/tasks.jpg" width=80% height=80% center>

**Abstract:** This paper introduces the Urban Socio-Semantic Segmentation dataset named SocioSeg, a new resource comprising satellite imagery, digital maps, and pixel-level labels of social semantic entities organized in a hierarchical structure. Additionally, we propose a novel vision-language reasoning framework called SocioReasoner that simulates the human process of identifying and annotating social semantic entities via cross-modal recognition and multi-stage reasoning. We employ reinforcement learning to optimize this non-differentiable process and elicit the reasoning capabilities of the vision-language model. Experiments demonstrate our approach's significant gains over state-of-the-art models and strong zero-shot generalization.


## 1. Installation
- OS: Linux distribution support for CUDA
- Hardware: At least 4x NVIDIA H20 (or A100 80GB) GPUs
- Framework: This repository is based on [ROLL](https://github.com/alibaba/ROLL), following the below installation instructions.
```bash
conda create -n socioseg python=3.10 -y
conda activate socioseg
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install 'transformer-engine[pytorch]==2.2.0' deepspeed==0.16.4 vllm==0.8.4 --no-build-isolation
pip install -e .
```

## 2. Dataset and Pretrained Model
Dataset: We provide the dataset in Huggingface: [SocioSeg](https://huggingface.co/datasets/vvangfaye/SocioSeg). We also provide raw dataset files from [Google Drive](https://drive.google.com/drive/folders/1j3Y6v1k1f1b3g7JH1F2KX13J3J3J3J3?usp=sharing).


Pretrained Model: We provide the pretrained SocioReasoner-3B model on Huggingface: [SocioReasoner-3B](https://huggingface.co/vvangfaye/SocioReasoner-3B).

## 3. Training

```bash
# default using the hugginface dataset
sh examples/train/train.sh
```
If you want to use the raw dataset files, please change the `actor_train.data_args.file_name` and `validation.data_args.file_name` in [examples/train/rlvr_megatron.yaml](examples/train/rlvr_megatron.yaml) to your local dataset path.

The trained model will be saved in `./output/train/checkpoint/`

## 4. Evaluation and Visualization

```bash
# default using the hugginface dataset and model
sh examples/infer/infer.sh
```
If you want to use the raw dataset files or the model trained by yourself, please change the `actor_train.data_args.file_name` and `pretrain` in [examples/infer/rlvr_megatron.yaml](examples/infer/rlvr_megatron.yaml)

The evaluation and visualization results will be saved in `./output/infer/result/`

<img src="./assets/results.jpg" width=90% height=90% center>

## Citation
```bibtex
@article{wang2026socioreasoner,
  title={Urban Socio-Semantic Segmentation with Vision-Language Reasoning}, 
  author={Yu Wang and Yi Wang and Rui Dai and Yujie Wang and Kaikui Liu and Xiangxiang Chu and Yansheng Li},
  journal={arXiv preprint arXiv:2601.10477},
  year={2026}
}
```

## Acknowledgements
We thank the authors of [ROLL](https://github.com/alibaba/ROLL) and [SegZero](https://github.com/JIA-Lab-research/Seg-Zero).
