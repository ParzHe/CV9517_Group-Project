# CV9517_Group-Project

The goal of this group project is to develop and compare different computer vision methods for segmenting standing dead trees in aerial images of forests.

## Project Structure

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

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ParzHe/CV9517_Group-Project.git
   cd CV9517_Group-Project
   ```

2. Set up the environment:

    If you have `conda` installed, you can create the environment using the provided `environment.yaml` file:

    ```shell
    conda env create -f environment.yaml
    ```

    If the pip installation is slow, you can comment the `pip` part in `environment.yaml` and use the `uv pip` command to install dependencies from `requirements.txt`:

    ```shell
    uv pip install -r requirements.txt
    ```

3. **Activate the environment**:

   ```shell
   conda activate CVers
   ```
