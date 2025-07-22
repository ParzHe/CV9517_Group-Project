# **COMP9517 Computer Vision 25T2 Group Project**

The goal of this group project is to develop and compare different computer vision methods for segmenting standing dead trees in aerial images of forests.

## 0. Project Zotero Library

This project has a Zotero library that contains references to papers, articles, and other resources relevant to the project. You can access the Zotero library at the following link:

[https://www.zotero.org/groups/6056458/cvers](https://www.zotero.org/groups/6056458/cvers)

## 1. Project Structure

TODO: Add more details about the project structure and its contents.

```plaintext
CV9517_Group-Project/
├── data/                    # Directory for storing datasets
├── notebooks/               # Directory for Jupyter notebooks
├── src/                     # Directory for source code
│   └── utils.py
├── environment.yaml         # Conda environment file for dependencies
└── requirements.txt         # Pip requirements file
```

---

## 2. Setup Environment Instructions

### 2.1 **Clone the repository**

   ```bash
   git clone https://github.com/ParzHe/CV9517_Group-Project.git
   cd CV9517_Group-Project
   ```

### 2.2 **Set up environment**

If you do not have `conda` installed, you can install it from the [Anaconda website](https://www.anaconda.com/download).

#### 2.2.1 Setup OpenMMLab Environment with Conda (for the most semantic segmentation)

- Create a `openmmlab` conda environment using the provided `environment.yaml` file:

   ```shell
   conda env create -f ./openmmlab/environment.yaml
   ```

- Activate the `openmmlab` environment:

   ```shell
   conda activate openmmlab
   ```

- Install `MMCV` using `mim`:

   ```shell
   mim install mmengine
   mim install "mmcv>=2.0.0"
   ```

- Install `MMSegmentation`:

   ```shell
   git config --system core.longpaths true
   git clone -b main https://github.com/open-mmlab/mmsegmentation.git
   cd mmsegmentation
   pip install -v -e .
   # '-v' means verbose, or more output
   # '-e' means installing a project in editable mode,
   # thus any local modifications made to the code will take effect without reinstallation. 
   ```

#### 2.2.2 Setup CVers Environment with Conda (for Pytorch Lightning, ADA-Net, and SAM2)

- Create a `CVers` conda environment using the provided `environment.yaml` file:

   ```shell
   # in the root directory of the project
   conda env create -f ./environment.yaml
   ```

### 3. **Activate the environmen**

Environments for different tasks can be activated using the following commands:

   ```shell
   conda activate openmmlab # For the MMSegmentation environment, this is could be more suitable for most semantic segmentation tasks.
   ```

   OR

   ```shell
   conda activate CVers # For the Pytorch Lightning environment. e.g. ADA-Net
   ```

---

## 3. Project Methods

### 3.1 U-Net

Person in Charge: To Be Determined

Refer to [GitHub repository](https://github.com/arbit3rr/UNet-AerialSegmentation)

Dev Environment:

- **Option 1**: `openmmlab` . You can learn [MMSegmentation](https://mmsegmentation.readthedocs.io/) and use it to train U-Net as you wish.
- **Option 2**: `CVers` . You can use the `pytorch_lightning` framework and [Segmentation Models PyTorch](https://smp.readthedocs.io/) to implement U-Net.

### 3.2 ADA-Net ( Be responsible for Replication Experiment )

Person in Charge: To Be Determined

Refer to the paper [ADA-Net：ADA-Net: Attention-Guided Domain Adaptation Network with Contrastive Learning for Standing Dead Tree Segmentation Using Aerial Imagery](https://arxiv.org/abs/2504.04271) and the [GitHub repository](https://github.com/meteahishali/ADA-Net)

Dev Environment: `CVers` （The `CVers` environment is almost the same to the ADA-Net GitHub repository [env.yml](https://github.com/meteahishali/ADA-Net/blob/main/env.yml)）

### 3.3 SAM2 （Zero-Shot Segmentation）

Person in Charge: [Zhen Yang（杨震）](https://github.com/DravenYiZ)

Refer to:

1. The [GitHub repository](https://github.com/facebookresearch/sam2)
2. [Enabling Meta’s SAM 2 model for Geospatial AI on satellite imagery](https://wherobots.com/blog/sam-2-model-geospatial-ai-satellite-imagery/)
3. [axXiv paper](https://arxiv.org/abs/2503.07266): "Customized SAM 2 for Referring Remote Sensing Image Segmentation"

Dev Environment: `CVers` or other environments as needed. When create `CVers` the SAM2 package will be installed automatically.

### 3.4 SAM2-UNet

Person in Charge: To Be Determined

Refer to the paper [SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation](https://arxiv.org/abs/2408.08870) and the [GitHub repository](https://github.com/WZH0120/SAM2-UNet)

Dev Environment: `CVers` or other environments as needed

### 3.5 SAM2-ADANet

Person in Charge: To Be Determined

This is a combination of SAM2 and ADA-Net, which is expected to achieve better performance than both methods. It is a new method proposed by our group. To be implemented.

Dev Environment: `CVers`
