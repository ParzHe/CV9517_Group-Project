# **COMP9517 Computer Vision 25T2 Group Project**

The goal of this group project is to develop and compare different computer vision methods for semantic segmentation of dead trees in aerial images. The dataset used for this project is [![Dataset](https://img.shields.io/badge/uses-Aerial_Imagery_for_Standing_Dead_Tree_Segmentation-orange?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/meteahishali/aerial-imagery-for-standing-dead-tree-segmentation), which contains RGB and NRG (NIR-RG) images.

This project is implemented using [![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org) and [![PyTorch Lightning](https://img.shields.io/badge/PyTorch_Lightning-2.5.2-792ee5?logo=lightning&logoColor=white&style=flat-square)](https://lightning.ai/docs/pytorch/stable/), and it includes over 60+ deep learning methods for semantic segmentation powered by [![Use](https://img.shields.io/badge/uses-segmentation__models.pytorch-blue?logo=github&style=flat-square)](https://github.com/qubvel-org/segmentation_models.pytorch). And, we also try zero-shot and fine-tuning with [![SAM2](https://img.shields.io/badge/uses-SAM2_by_Meta-black?logo=github&logoColor=white&style=flat-square)](https://github.com/facebookresearch/sam2) .

[TOC]

## **0. Project Zotero Library**

This project has a Zotero library that contains references to papers, articles, and other resources relevant to the project. You can access the Zotero library at the following link:

[https://www.zotero.org/groups/6056458/cvers](https://www.zotero.org/groups/6056458/cvers)

## **1. ProjectCode Structure**

This project is organized into several directories and files, each serving a specific purpose. Below is an overview of the project structure:

```plaintext
CV9517_Group-Project/
├── checkpoints/                 # Directory for model checkpoints (Local storage)
├── data/                        # Directory for data processing scripts
│   ├── __init__.py              # Init file for data processing module
│   ├── datamodule.py            # Script for lightning datamodule (Supports merged, RGB, and NRG modalities)
│   ├── dataset.py               # Script for dataset class
│   ├── transforms.py            # Script for data transformations
│   └── utils.py                 # Utility functions for data processing
├── datasplits                   # Directory for data splits csv files
│   └── data_split_42_70_10.csv  # seed 42, train 70%, val 10%, test 20%
├── gradio/                      # Directory for Gradio app scripts (Future work)
│   └── app.py                   # Script for Gradio app (Future work)
├── lighting_modules             # Directory for lightning modules
│   ├── __init__.py              # Init file for lightning modules
│   ├── sam2_module              # Script for SAM2 lightning module (Future work)
│   ├── segmentation_module      # Base lightning module for this dead tree segmentation project
│   ├── smp_module               # Script for Segmentation Models PyTorch lightning module
│   └── u2net_module             # Script for U2Net lightning module
├── logs/                        # Directory for logs (Local storage, e.g., TensorBoard logs)
├── models/                      # Directory for model scripts
│   ├── __init__.py              # Init file for models module
│   ├── smp_models_util.py       # Utility functions for models of Segmentation Models PyTorch
│   └── u2net.py                 # Script for U2Net model
├── notebooks/                   # Directory for Jupyter notebooks (It is for simple demonstration and testing)
│   └─ segmentation_models.ipynb # Notebook for testing Segmentation Models PyTorch
├── outputs/                     # Directory for output files (e.g., model predictions, evaluation logs)
├── sam2/                        # Directory for SAM2 (You should clone files from the SAM2 official GitHub repository)
├── scripts/                     # Directory for scripts
│   ├── sam2_fine_tune.py        # Script for fine-tuning SAM2 model
│   ├── sam2_ft_lightning.py     # Script for fine-tuning SAM2 model using PyTorch Lightning (Future work)
│   ├── sam2_zero_shot.py        # Script for test zero-shot segmentation using SAM2
│   ├── test_sam2_ft.py          # Script for testing fine-tuned SAM2 model
│   ├── test_smp.py              # Script for testing Segmentation Models PyTorch model
│   ├── train_smp.py             # Script for training Segmentation Models PyTorch model (Over 60 models, with different architectures & feature extractors, on 3 modalities)
│   └── train_u2net.py           # Script for training U2Net model (Future work)
├── utils/                       # Directory for utility scripts
│   ├── __init__.py              # Init file for utils module
│   ├── callbacks.py             # Script for abstracting lightning trainer callbacks
│   ├── logger.py                # Script for abstracting logger construction for each kind of activity
│   └── paths.py                 # Script for paths management
├── .gitignore                   # Git ignore file to exclude unnecessary files from version control
├── environment.yaml             # Conda environment file for dependencies
├── README.md                    # Project overview and instructions
└── requirements.txt             # Pip requirements file
```

---

## **2. Setup Environment Instructions**

### 2.1 Clone the repository

   ```bash
   git clone https://github.com/ParzHe/CV9517_Group-Project.git
   cd CV9517_Group-Project
   ```

### 2.2 Set up environment

We suggest you to use Linux for the trying. If you are using Windows, we suggest you use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux).

If you do not have `conda` installed, you can install it from the [Anaconda website](https://www.anaconda.com/download).

#### 2.2.1 Setup CVers Environment with Conda (for Pytorch Lightning, ADA-Net, and SAM2)

- Create a `CVers` conda environment using the provided `environment.yaml` file:

   ```shell
   # in the root directory of the project
   conda env create -f ./environment.yaml
   ```

- Activate the `CVers` environment:

   ```shell
   conda activate CVers
   ```

#### 2.2.2 Setup for SAM2 (Optional)
- Install `sam2`:

   ```shell
   git clone https://github.com/facebookresearch/sam2.git && cd sam2

   pip install -e .

   ```

- Download the SAM2 model weights with the following command:

   ```shell
   cd sam2/checkpoints
   bash download_ckpts.sh
   ```

- Then back to the root directory of the project:

   ```shell
   cd ../..
   ```

### 2.3 **Activate the environment**

Our environment can be activated using the following commands:

   ```shell
   conda activate CVers # For the Pytorch Lightning environment.
   ```
---

## **3. Project Methods**

### 3.1 SMP (Segmentation Models PyTorch)

**Person in Charge**: [Lintao He 何林涛](https://github.com/ParzHe)

**Participants (listed in alphabetical order by name)**: [Bowei Cheng 程柏威](https://github.com/Awzhcc), [Chencan Que 阙晨灿](https://github.com/IkarosForever), [Zhen Yang 杨震](https://github.com/DravenYiZ), [Zitong Wei 魏子童](https://github.com/WIZ-ROC)

You can find the papers by directly clicking the architecture and feature extractor names below.

#### Applied Architectures

- **[Unet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)**
- **[Unet++](https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1)**: Do not support `Mix Vision Transformer (MixViT)` as the feature extractor (encoder).
- **[Linknet](https://ieeexplore.ieee.org/document/8305148)**: Do not support `Mix Vision Transformer (MixViT)` as the feature extractor (encoder).
- **[FPN](https://ieeexplore.ieee.org/document/8099589)**
- **[PSPNet](https://ieeexplore.ieee.org/document/8100143)**
- **[PAN](https://arxiv.org/abs/1805.10180)**: Do not support `DenseNet` as the feature extractor (encoder).
- **[DeepLabV3](https://arxiv.org/abs/1706.05587)**: Do not support `DenseNet` as the feature extractor (encoder).
- **[DeepLabV3+](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_4)**: Do not support `DenseNet` as the feature extractor (encoder).
- **[UperNet](https://link.springer.com/chapter/10.1007/978-3-030-01228-1_26)**
- **[SegFormer](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)**

#### Applied Feature Extractors (Encoders)

Select the following feature extractors (encoders) because they have close parameters.

- **[DenseNet161](https://ieeexplore.ieee.org/document/8099726)**: Do not support  `DeepLabV3`, `DeepLabV3+` and `PAN` architectures.
- **[EfficientNet-B5](https://proceedings.mlr.press/v97/tan19a.html?ref=%2A%2Aa-ai-gmbh.ghost.io)**
- **[Mix Vision Transformer (MixViT)](https://www.sciencedirect.com/science/article/pii/S0031320322006112)**: Do not support `Unet++` and `Linknet` architectures.
- **[ResNet50](https://ieeexplore.ieee.org/document/7780459)**
- **[ResNeXt50_32x4d](https://ieeexplore.ieee.org/document/8100117)**
- **[SE-ResNet50 and SE-ResNeXt50_32x4d](https://ieeexplore.ieee.org/document/8578843)**

### 3.2 SAM2 （Zero-Shot Segmentation & Fine-Tuning）

**Person in Charge**: [Zhen Yang（杨震）](https://github.com/DravenYiZ)

**Participant**: [Lintao He 何林涛](https://github.com/ParzHe)

Refer to:

1. The [GitHub repository](https://github.com/facebookresearch/sam2)
2. [Enabling Meta’s SAM 2 model for Geospatial AI on satellite imagery](https://wherobots.com/blog/sam-2-model-geospatial-ai-satellite-imagery/)
3. [axXiv paper](https://arxiv.org/abs/2503.07266): "Customized SAM 2 for Referring Remote Sensing Image Segmentation"
4. [axXiv paper](https://arxiv.org/abs/2506.03114): "Zero-Shot Tree Detection and Segmentation from Aerial Forest Imagery"

Dev Environment: `CVers` or other environments as needed 

> [!NOTE]
> You need to manually clone the SAM2 repository and download the model weights as described in [2.2.2 Subsection](#222-setup-for-sam2-optional22-setup-cvers-environment-with-conda-for-pytorch-lightning-and-ada-net-and-sam2).

## **4. Training & Testing**

### 4.1 Dataset & Pre-trained Weights

The code will automatically download the dataset from Kaggle, with the `kagglehub` package. If you want to delete the dataset, you need to go to the `.cache/kagglehub/datasets` directory. And, the datasplit csv files are stored in the `datasplits` directory. You can set different seeds and split ratios by modifying the parameters of `AerialDeadTreeSegDataModule` in different training or test scripts.

For Segmentation Models PyTorch, the pre-trained weights will be automatically downloaded. If you want to delete the pre-trained weights, you need to go to the `~/.cache/huggingface/hub` directory to delete as you wish.

For SAM2, the pre-trained weights should download manually as described in the [2.2.2 Subsection](#222-setup-for-sam2-optional22-setup-cvers-environment-with-conda-for-pytorch-lightning-and-ada-net-and-sam2) section.

### 4.2 Training

#### 4.2.1 SMP (Segmentation Models PyTorch)

To train the Segmentation Models PyTorch, you can use the following command in the root directory of the project:

```bash
conda activate CVers  # Activate the CVers environment
python scripts/train_smp.py
```

This will train all the architectures and feature extractors specified in the `arch_list` and `encoder_only` variables in the `scripts/train_smp.py` file. The training will be performed on the merged, RGB and NRG modalities by default.

**4.2.1.1 Train on a specific architecture or feature extractor:**

If you want to train a specific architecture or feature extractor, you can modify the `arch_list` and `encoder_only` variables in the `scripts/train_smp.py` file. For example, to train the `Unet` architecture with `DenseNet161` as the feature extractor, you can set:

```python
arch_list = ['Unet']
encoder_only = ['densenet161']
```

**4.2.1.2 Train on a specific modality:**

If you want to train on a specific modality, you can modify the `modality_list` variable in the `scripts/train_smp.py` file. For example, to train only on the RGB modality, you can set:

```python
modality_list = ['rgb']
```

**4.2.1.3 Loss Function:**

The default loss function is combination of `JaccardLoss` and `FocalLoss`. You can modify the loss function in the `scripts/train_smp.py` file by changing the `LOSS1` and `LOSS2` variables. For example, to use only `JaccardLoss`, you can set:

```python
LOSS1 = 'JaccardLoss'
LOSS2 = None
```

**4.2.1.4 Early Stopping:**

The training script will automatically stop if the validation loss does not improve for 30 epochs. This can avoid overfitting and save training time. You can modify the `EARLY_STOP_PATIENCE` variable in the `scripts/train_smp.py` file to change the patience value.

**4.2.1.5 Checkpoints Saving:**
The training script will automatically save the best 2 model checkpoints in the `checkpoints/` directory, each models have 3 modalities (merged, RGB, NRG) folders to save the checkpoints. The checkpoints will be saved with the following naming format:

```plaintext
smp_{encoder}_{arch}/{modality}_{target size}_{version suffix}/{epoch}-{val per_image_mIou}.ckpt
```

Where:
- `{encoder}`: The feature extractor (encoder) name.
- `{arch}`: The architecture name.
- `{modality}`: The modality name (merged, rgb, nrg).
- `{target size}`: The target size of the input images.
- `{version suffix}`: The version suffix, can be changed as needed.
- `{epoch}`: The epoch number of the saving model.
- `{val per_image_mIou}`: The validation imagewise mean Intersection over Union (mIoU) value.


>[!NOTE]
> 1. The training script will automatically search the suggested learning rate using the `find_lr` method from the `lightning` library.
> 2. The training script will automatically log the training and validation metrics to the `logs/` directory, which can be viewed using TensorBoard. And, logging summary will also be done in the certain `checkpoints/` directory.

> [!TIP]
> You can modify the `BATCH_SIZE` and `ACCUMULATE_GRAD_BATCHES` variables in the `scripts/train_smp.py` file to adjust the batch size and gradient accumulation. The default values are `BATCH_SIZE = 32` and `ACCUMULATE_GRAD_BATCHES = 1`. If you encounter out-of-memory (OOM) errors, you can try reducing the batch size and increasing the gradient accumulation.

#### 4.2.2 SAM2 (Fine-Tuning)
To fine-tune the SAM2 model, you can use the following command in the root directory of the project:

```bash
conda activate CVers  # Activate the CVers environment
python scripts/sam2_fine_tune.py
```

### 4.3 Testing

#### 4.3.1 SMP (Segmentation Models PyTorch)

To test the Segmentation Models PyTorch, you can use the following command:

```bash
conda activate CVers  # Activate the CVers environment
python scripts/test_smp.py
```

This script will automatically load the best model checkpoints from the `checkpoints/` directory and perform inference on the test dataset split with the specified split and modality. The results will generate a csv file in the `outputs/smp_test_results` directory.

#### 4.3.2 SAM2 (Zero-Shot Segmentation & Fine-Tuning)

**4.3.2.1 Zero-Shot Segmentation**

For zero-shot segmentation using SAM2, you can use the following command:

```bash
conda activate CVers  # Activate the CVers environment
python scripts/sam2_zero_shot.py
```

This script will perform zero-shot segmentation using the SAM2 model on the test dataset split. The results will be saved in the `outputs/sam2_zs_inference` directory.

**4.3.2.2 Fine-Tuning Test**

For fine-tuning the SAM2 model, you can use the following command:

```bash
conda activate CVers  # Activate the CVers environment
python scripts/test_sam2_ft.py
```

This script will load the fine-tuned SAM2 model and perform inference on the test dataset split. The results will be saved in the `outputs/sam2_ft_inference` directory.

## **5. Future Work**

We plan to implement the following features in the future:

- **Gradio App**: We will develop a Gradio app to demonstrate the segmentation results of the trained models. The app will allow users to upload images and visualize the segmentation results in real-time.
- **More Models**: We will add more models to the project, including U2Net and other segmentation models. This will help us to compare the performance of different models on the same dataset.
- **SAM2 zero-shot with prompt**: We will explore the use of prompt techniques to improve the zero-shot segmentation capabilities of the SAM2 model in this task.
