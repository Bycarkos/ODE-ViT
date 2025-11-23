### 🔥 [*ODE-ViT: Plug & Play Attention Layer from the Generalization of the ViT as an Ordinary Differential Equation*](https://arxiv.org/abs/2511.16501) 🔥

**Authors:**
[**Carlos Boned**](https://scholar.google.com/citations?user=FCb2rY8AAAAJ&hl=en),
[**David Romero**](),
[**Oriol Ramos**](https://scholar.google.com/citations?user=4Dvggx4AAAAJ&hl=en),

### [![arXiv](https://img.shields.io/badge/arXiv-2511.16501-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.16501) ![Visits](https://img.shields.io/badge/Visits-118-blue)

<img width="4502" height="1510" alt="Architectures (3)" src="https://github.com/user-attachments/assets/8a5cd02d-fd4e-4680-8600-c8efa29039bd" />


## 📖 Abstract
In recent years, increasingly large models have achieved outstanding performance across CV tasks. However, these models demand substantial computational resources and storage, and their growing complexity limits our understanding of how they make decisions. Most of these architectures rely on the attention mechanism within Transformer-based designs. Building upon the connection between residual neural networks and ordinary differential equations (ODEs), we introduce ODE-ViT, a Vision Transformer reformulated as an ODE system that satisfies the conditions for well-posed and stable dynamics. Experiments on CIFAR-10 and CIFAR-100 demonstrate that ODE-ViT achieves stable, interpretable, and competitive performance with up to one order of magnitude fewer parameters, surpassing prior ODE-based Transformer approaches in classification tasks. We further propose a plug-and-play teacher-student framework in which a discrete ViT guides the continuous trajectory of ODE-ViT by treating the intermediate representations of the teacher as solutions of the ODE. This strategy improves performance by more than 10% compared to training a free ODE-ViT from scratch.


## 📣 Latest News 📣
- **`25 November 2025`** Our survey paper have dropped in [arXiv](https://arxiv.org/abs/2511.16501) !!



## Overview

* **ODE-ViT architecture**: We introduce a ViT-based formulation in which the attention block is reformulated as an ODE that satisfies the conditions for stability and wellposedness.

* **Teacher–student training**: We propose a teacher–student framework in which a discrete ViT supervises the continuous trajectory of the ODE-ViT, aligning its intermediate states with the teacher’s representations. This framework operates in a self-supervised manner and is specifically designed to replace the attention encoder
of the ViT with our ODE-based formulation, making the proposed module fully  plug-and-play.


## Teacher-Student Framework

<img width="4505" height="1683" alt="Architectures (2)" src="https://github.com/user-attachments/assets/ef027821-ea73-49ad-8576-51ded474b967" />

[![](https://img.shields.io/badge/Teacher_Model-repo-blue?logo=Deno)](https://github.com/facebookresearch/dino)

## Qualitative Results


### Watch Video 🔥🔥
![lion](https://github.com/user-attachments/assets/4bbfb471-186f-4d7f-84f1-e1dd7e2314a1)

### And More Results !!

<img width="4553" height="4039" alt="Architectures (1)" src="https://github.com/user-attachments/assets/405c3e15-c73f-443d-82d2-b26f976a8a68" />


## Results

<div align="center">

| Model | Dataset | Params (M) | Losses | Acc@1 | Acc@3 | Acc@5 |
|-------|---------|------------|--------|-------|-------|-------|
| **Teacher – DINO-base + Training Only Head** | Cifar10 | 85 | CE | 0.923 | 0.993 | 0.997 |
| | Cifar100 | 85 | CE | 0.881 | 0.968 | 0.982 |
| | Imagenet100 | 85 | CE | 0.923 | 0.981 | 0.990 |
| **ODE (Zhong et al., 2022)** | Cifar100 | 0.7 | CE | 0.533 | - | - |
| **ODE** | Cifar10 | 0.5 | CE | 0.809 | 0.980 | 0.990 |
| | Cifar100 | 4.2 | CE | 0.579 | 0.728 | 0.794 |
| | Imagenet100 | 7 | CE | 0.513 | 0.701 | 0.754 |
| **ODE Teacher–Student Base** | Cifar10 | 7 | MSE+JasMin | 0.885 | 0.980 | 0.992 |
| | Cifar100 | 7 | MSE+JasMin+CE | 0.721 | 0.872 | 0.914 |
| | Imagenet100 | 7 | MSE+JasMin | 0.684 | 0.817 | 0.865 |
| **ODE Teacher–Student Small** | Cifar10 | 3.8 | MSE+JasMin | 0.867 | 0.973 | 0.991 |
| | Cifar100 | 3.8 | MSE+JasMin | 0.657 | 0.819 | 0.914 |

</div>



# Training and Inference (Future Release of HuggingFace ODE-ViT Layer)

## Configuration Files
The configuration files are based on hydra

<a href="https://hydra.cc/docs/intro/">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSPkr8HAexNN5J_x9SeVtVz3xyyDTEk3RBJJQ&s" alt="Hydra Docs" width="100"/>
</a>


# Star History ⭐
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=Bycarkos/ODE-ViT&type=Date
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=Bycarkos/ODE-ViT&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=Bycarkos/ODE-ViT&type=Date"
  />
</picture>
