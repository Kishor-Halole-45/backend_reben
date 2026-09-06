# Training Script for BigEarthNet v2.0 (reBEN)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10891137.svg)](https://doi.org/10.5281/zenodo.10891137)
[![arXiv](https://img.shields.io/badge/arXiv-2407.03653-b31b1b.svg)][arxiv]

[TU Berlin](https://www.tu.berlin/) | [RSiM](https://rsim.berlin/) | [DIMA](https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/) | [BigEarth](http://www.bigearth.eu/) | [BIFOLD](https://bifold.berlin/)
:---:|:---:|:---:|:---:|:---:
<a href="https://www.tu.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/tu-berlin-logo-long-red.svg" width=150em></a> |  <a href="https://rsim.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/RSiM_Logo_1.png" alt="RSiM Logo" width=100em></a> | <a href="https://www.dima.tu-berlin.de/menue/database_systems_and_information_management_group/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/DIMA.png" width=50em height=50em alt="DIMA Logo"></a> | <a href="http://www.bigearth.eu/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BigEarth.png" alt="BigEarth Logo" width=150em></a> | <a href="https://bifold.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BIFOLD_Logo_farbig.png" alt="BIFOLD Logo" width=150em></a>

![[BigEarthNet](http://bigearth.net/)](https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/combined_2000_600_2020_0_wide.jpg)

## Pre-requisites

## Grounding web app

The repository now includes a React interface for query-driven object grounding. Upload a satellite image, type a request such as `buildings` or `roads and water`, and the OWLv2 detector returns matching rectangles over the image.

Install the API dependencies with Poetry, then start the backend:

```bash
poetry install
poetry run uvicorn api_server:app --reload --port 8000
```

In a second terminal, start the React client:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The first grounding request downloads the OWLv2 checkpoint from Hugging Face and caches it locally. GPU is used automatically when CUDA is available.

We recommend using the pyproject.toml file to install the required dependencies based on
[Poetry](https://python-poetry.org/). For use with Poetry, CUDA 12.x is required. To use CUDA 11.x, the commented lines
in the pyproject.toml file have to be uncommented. To install the dependencies using Poetry, run `poetry install`.

Otherwise, the following dependencies are required:

- Python 3.9 - 3.12 (e.g. install with `sudo apt install python3.10`)
- configilm[full] (e.g. install with `pip install configilm[full]~=0.7.0`) 0.7.0 or higher
- wandb (e.g. install with `pip install wandb`)
- numpy 1.x (e.g. install with `pip install numpy~=1.26.4`)

Please also create an account on [wandb](https://wandb.ai/) and login using `wandb login`. This is required to log the
training progress.

## Data

The data can be downloaded from the [BigEarthNet website](http://bigearth.net/). Extract the data to a folder and create
an [LMDB](https://lmdb.readthedocs.io/en/release/) database using
the [rico-hdl](https://github.com/kai-tub/rico-hdl) tool.
The tool can be downloaded from the [BigEarthNet website](http://bigearth.net/) or installed as an [AppImage](www.appimage.org) from the
[rico-hdl releases page](https://github.com/kai-tub/rico-hdl/releases/latest) or as an docker image from the [GitHub container registry](https://github.com/kai-tub/rico-hdl/pkgs/container/rico-hdl).

Enter the paths to the following files in the `scripts/train_BigEarthNetv2_0.py` script:

- `images_lmdb` (path to the LMDB database)
- `metadata_parquet` (path to the metadata file)
- `metadata_snow_cloud_parquet` (path to the metadata file with snow and cloud patches, not strictly necessary)

Enter the paths at the top of the `scripts/train_BigEarthNetv2_0.py` script in the `BENv2_DIR_DICT` dictionary.

## Training

Run the training script with `python scripts/train_BigEarthNetv2_0.py`. 
The script will train the model and log the progress to wandb.

The following parameters can be adjusted as arguments to the script:

- `--architectures` (default: `resnet18`) The architectures to train. Many architectures from the `timm` library are
  supported.
- `--seed` (default: `42`) The seed to use for the random number generators.
- `--lr` (default: `1e-3`) The learning rate to use for training for the optimizers `AdamW`.
- `--epochs` (default: `100`) The number of epochs to train for.
- `--bs` (default: `16`) The batch size to use for training.
- `--drop_rate` (default: `0.375`) The dropout rate to use for the models.
- `--drop_path_rate` (default: `0.0`) The drop path rate to use for the models.
- `--workers` (default: `8`) The number of workers to use for the data loader.
- `--bandconfig` (default: 'all') The band configuration* to use. The following configurations are supported:
    - `all` 10m and 20m bands from Sentinel-2 and all bands from Sentinel-1 (12 bands in total)
    - `s2` 10m and 20m bands from Sentinel-2 (10 bands in total)
    - `s1` all bands from Sentinel-1 (2 bands in total)
    - `all_full` 10m, 20m, and 60m bands from Sentinel-2 and all bands from Sentinel-1 (14 bands in total)
    - `s2_full` 10m, 20m, and 60m bands from Sentinel-2 (12 bands in total)
    - `s1_full` all bands from Sentinel-1 (2 bands in total, same as `s1`)
- `--use-wandb` or `--no-use-wandb` (default: `False`, same as `--no-use-wandb`) Whether to use wandb for logging. If
  `--no-use-wandb` is set, the script will not log to wandb but checkpoints are still saved and metrics are printed to
  the console.
- `--upload-to-hub` or `--no-upload-to-hub` (default: `False`, same as `--no-upload-to-hub`) Whether to upload the model
  to the Hugging Face model hub. If `--upload-to-hub` is set, the model will be uploaded to the Hugging Face model hub
  after training. Note that you have to be logged in to the Hugging Face model hub using `huggingface-cli login` for
  this to work. For this you need a Hugging Face account and a token which can be obtained from the Hugging Face
  website.
- `--test-run` or `--no-test-run` (default: `True`) Whether to only run a few batches for testing. If `--no-test-run` is
  set, the full dataset will be used for training and testing.
- `--hf-entity` (default: `None`) The Hugging Face entity to use for uploading the model. Has to be set if
  `--upload-to-hub` is set.

The full command to train the resnet50 model as described in the BigEarthNet v2.0 paper with all bands from Sentinel-2 
is as follows:

```bash
python train_BigEarthNetv2_0.py --no-test-run --use-wandb --upload-to-hub --architecture=resnet50 --bandconfig=s2 --bs=512 --lr=0.001
```

This command
 - uses the full dataset instead of only a few batches to test
 - logs the training progress to wandb
 - uploads the model to the Hugging Face model hub after training and testing
 - uses the resnet50 architecture
 - uses only the Sentinel-2 bands
 - uses a batch size of 512
 - uses a learning rate of 0.001

The trained model weights are saved in the `models` directory and on huggingface. To load the model with the corresponding weights run:

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

model = BigEarthNetv2_0_ImageClassifier.from_pretrained("<entity>/<model-name>")
```
e.g.

```python
from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

model = BigEarthNetv2_0_ImageClassifier.from_pretrained("BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.1.1")
```

> Note:
> The model configuration is automatically loaded from the Hugging Face model hub.

> Note:
> Not all S2 bands are used to train and evaluate the models. For details, please refer to the
[BigEarthNet v2.0 paper][arxiv].

If you use this work, please cite:

> K. Clasen, L. Hackel, T. Burgert, G. Sumbul, B. Demir, V. Markl, "[reBEN: Refined BigEarthNet Dataset for Remote Sensing Image Analysis][arxiv]", IEEE International Geoscience and Remote Sensing Symposium (IGARSS), 2025.


```bibtex
@inproceedings{clasen2025refinedbigearthnet,
  title={{reBEN}: Refined BigEarthNet Dataset for Remote Sensing Image Analysis},
  author={Clasen, Kai Norman and Hackel, Leonard and Burgert, Tom and Sumbul, Gencer and Demir, Beg{\"u}m and Markl, Volker},
  year={2025},
  booktitle={IEEE International Geoscience and Remote Sensing Symposium (IGARSS)},
}
```

[arxiv]: https://arxiv.org/abs/2407.03653

> L. Hackel, K. Clasen, B. Demir, "ConfigILM: A General Purpose Configurable Library for Combining Image and Language Models for Visual Question Answering.", SoftwareX 26 (2024): 101731.

```bibtex
@article{hackel2024configilm,
  title={ConfigILM: A general purpose configurable library for combining image and language models for visual question answering},
  author={Hackel, Leonard and Clasen, Kai Norman and Demir, Beg{\"u}m},
  journal={SoftwareX},
  volume={26},
  pages={101731},
  year={2024},
  publisher={Elsevier}
}
```
