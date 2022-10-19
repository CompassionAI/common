# CompassionAI common repository

Common utilities and dataset preparation tools in use by various CompassionAI projects.

## Installation

There are two modes for this library - inference and research. We provide instructions for Linux.

 - Inference should work on MacOS and Windows _mutatis mutandis_.
 - We *very strongly* recommend doing research *only* on Linux. We will not provide any support to people trying to perform research tasks without installing Linux.

### Virtual environment

We strongly recommend using a virtual environment for all your Python package installations, including anything from CompassionAI. To facilitate this, we provide a simple Conda environment YAML file. We recommend first installing miniconda, see <https://docs.conda.io/en/main/miniconda.html>. We then recommend installing Mamba, see <https://github.com/mamba-org/mamba>.

```bash
bash Miniconda3-latest-Linux-x86_64.sh
conda install mamba -c conda-forge
mamba env create -f env-minimal.yml -n my-env
conda activate my-env
```

### Inference

Just install with pip:

```bash
pip install compassionai-common
```

### Research

Begin by installing for inference. Then install the CompassionAI data registry repo and set two environment variables:

```bash
$CAI_TEMP_PATH
$CAI_DATA_BASE_PATH
```

We strongly recommend setting them with conda in your virtual environment:

```bash
conda activate my-env
conda env config vars set CAI_TEMP_PATH=#directory on a mountpoint with plenty of space, does not need to be fast
conda env config vars set CAI_DATA_BASE_PATH=#absolute path to the CompassionAI data registry
```

Our code uses these environment variables to load datasets from the registry, output processed datasets and store training results.

You probably also want to install CUDA and PyTorch (>=1.12) with CUDA support - follow the instructions here <https://pytorch.org/get-started/locally/>. You don't need torchvision or torchaudio but it is safe to install them if you like. You can reinstall CUDA-enabled PyTorch with `pip` in your conda environment after installing everything as above.

For fine-tuning, you will need a powerful NVidia GPU. A GTX 1080 might work. We recommend at least an RTX 3080 Ti in a home setup, or a V100 if using a cloud. We have not tested non-NVidia GPUs.

For pre-training, you will need a TPU on GCP. We do not recommend fine-tuning on GPUs. We do not expect it to work on anything less than a DGX-2 or a p3dn.24xlarge instance.

## Usage

### Inference

This is a supporting library for our main inference repos, such as Lotsawa. You shouldn't need to use it directly.

### Research

This library contains components that are common to the various tasks performed by the other libraries, such as Manas and Garland.

 - Implements data loader objects such as KangyurLoader, TengyurLoader and TibetanDict.
 - Implements common PyTorch dataset objects, such as TokenTagDataset.
 - Provides utility functions for models and data, such as Hydra-Huggingface adapters, PyTorch callbacks, model downloaders and configuration providers.