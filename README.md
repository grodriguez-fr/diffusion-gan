# Diffusion-GAN Implementation

PyTorch implementation of Diffusion-GAN from scratch for the Deep Learning 2026 course project.

## Requirements

- Python 3.11 - 3.13 (3.13 recommended, latest stable as of Jan 2026)
- CUDA 12.6, 12.8, or 13.0 (for GPU support)
- Windows 10/11

## Setup

### 1. Install Python 3.13

Download and install Python 3.13 from [python.org](https://www.python.org/downloads/).

Latest stable: Python 3.13.11 (Dec 2025) or Python 3.14.2 (Dec 2025, bugfix phase)

### 2. Create Virtual Environment

```powershell
python -3.13 -m venv venv
```

### 3. Activate Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

If you encounter an execution policy error, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Note: PyTorch 2.9.1 (stable, Jan 2026) requires Python 3.10 or later.

### 4. Install Dependencies

```powershell
py -m pip install --upgrade pip
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

### 5. Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
diffusion-gan/
├── tools/
│   └── data_loader.py          # Dataset utilities
├── dataset_exploration.ipynb    # Dataset visualization
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Usage

### Explore Datasets

Open `dataset_exploration.ipynb` in Jupyter or VS Code to visualize MNIST, Fashion-MNIST, or CIFAR-10.

### Available Datasets

- MNIST (28x28 grayscale, 10 classes)
- Fashion-MNIST (28x28 grayscale, 10 classes)
- CIFAR-10 (32x32 RGB, 10 classes)

## Development

To deactivate the virtual environment:

```powershell
deactivate
```

## Citation

Based on the paper: "Diffusion-GAN: Training GANs with Diffusion" by Wang et al.


Si tu veux être très précis sur l'origine des blocs "ResBlock" (l'idée d'ajouter x + h à la fin), tu peux aussi citer les inventeurs du ResNet :

    Titre : Deep Residual Learning for Image Recognition

    Auteurs : Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    Année : 2016 (CVPR)

    Pourquoi ? C'est eux qui ont prouvé que faire des "Skip Connections" (le raccourci) permettait d'entraîner des réseaux très profonds