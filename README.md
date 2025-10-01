# AEROMamba

## About 

Official PyTorch implementation of 

**AEROMamba: An efficient architecture for audio super-resolution using generative adversarial networks and state space models**

whose demo is available in our [Webpage](https://aeromamba-super-resolution.github.io/).  Our model is closely related to [AERO](https://github.com/slp-rl/aero) and [Mamba](https://github.com/state-spaces/mamba), so make sure to check them out if any questions arise regarding these modules.

## 🆕 PyTorch 2.x Update

This repository has been upgraded to support **PyTorch 2.x** and **CUDA 12.x** for improved performance and compatibility with modern GPUs:

- ✅ **10-20% faster training** on RTX 30xx/40xx with TF32
- ✅ **Better memory efficiency** with PyTorch 2.x optimizations
- ✅ **Enhanced multi-GPU training** with improved DDP
- ✅ **Easier installation** with simplified dependencies
- ✅ **Full backward compatibility** with existing checkpoints

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed upgrade information and [CHANGELOG.md](CHANGELOG.md) for all changes.

## Installation

Requirements:
- Python 3.10.0 or later (tested with 3.10-3.11)
- PyTorch 2.0.0 or later
- CUDA 12.x

Instructions:
- Create a conda environment or venv with python>=3.10.0 
- Run `pip install -r requirements.txt`

### Manual Installation (if needed)

If there is any error in the previous step, make sure to install manually the required libs.

**For PyTorch with CUDA 12.x:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For Mamba and Causal Conv1D:**
```bash
pip install causal-conv1d>=1.1.0
pip install mamba-ssm>=1.1.0
```

**Note:** The environment variables for building causal-conv1d are no longer required with PyTorch 2.x. If you still encounter build issues, you can try:
```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

**Note for Mamba:** With PyTorch 2.x, Mamba is now installed via pip and no longer requires manual folder extraction. If you encounter issues, see MIGRATION_GUIDE.md for troubleshooting.

### Quick Install (Recommended)

We provide installation scripts for easy setup:

**Windows (PowerShell):**
```powershell
.\install.ps1
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

The scripts will:
- Verify Python version
- Check CUDA/GPU availability
- Guide you through installation options
- Verify the installation

### ViSQOL

We did not use ViSQOL for training and validation, but if you want to, see [AERO](https://github.com/slp-rl/aero) for instructions. 

## Datasets

### Download data

For popular music we use the mixture tracks of [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html#musdb18-hq-uncompressed-wav) dataset.

For piano music, we collected a private dataset from CDs whose metadata are described in our [Webpage](https://aeromamba-super-resolution.github.io/).

### Resample data

Data are a collection of high/low resolution pairs. Corresponding high and low resolution signals should be in different folders, eg: hr_dataset and lr_dataset. 

In order to create each folder, one should run `resample_data` a total of 5 times,
to include all source/target pairs.

We downsample once to a target 11.025 kHz, from the original 44.1 kHz.

e.g. for 11.025 and 44.1 kHz: \
`python data_prep/resample_data.py --data_dir <path for 44.1 kHz data> --out_dir <path for 11.025 kHz data> --target_sr 11025`

### Create egs files

For each low and high resolution pair, one should create "egs files" twice: for low and high resolution.  
`create_meta_files.py` creates a pair of train and val "egs files", each under its respective folder.
Each "egs file" contains meta information about the signals: paths and signal lengths.

`python data_prep/create_meta_files.py <path for 11.025 kHz data> egs/musdb/ lr` 

`python data_prep/create_meta_files.py <path for 44.1 kHz data> egs/musdb/ hr`

## Train

Run `train.py` with `dset` and `experiment` parameters, or set the default values in main_config.yaml file.  

`
python train.py dset=<dset-name> experiment=<experiment-name>
`

To train with multiple GPUs, run with parameter `ddp=true`. e.g.
`
python train.py dset=<dset-name> experiment=<experiment-name> ddp=true
`

## Test (on whole dataset)

`
python test.py dset=<dset-name> experiment=<experiment-name>
`

## Inference

### Single sample

`
python predict.py dset=<dset-name> experiment=<experiment-name> +filename=<absolute path to input file> +output=<absolute path to output directory>
`

### Multiple samples

`
bash predict_batch.sh <input_folder> <output_folder>
`

We also provide predict_with_ola.py to predict large files that do not fit in the GPU, without the need for segmentation, using Overlap-and-Add. The original predict.py is also capable of joining predicted segments, but its naïve method causes clicks. 

`
python predict_with_ola.py dset=<dset-name> experiment=<experiment-name> +folder_path=<absolute path to input folder> +output=<absolute path to output directory>
`
### Checkpoints

To use pre-trained models for MUSDB18-HQ or PianoEval data, one can download checkpoints from [here](https://poliufrjbr-my.sharepoint.com/:f:/g/personal/abreu_engcb_poli_ufrj_br/EhqOtFGTmeZNr-WNv976Jw8BLfpgBYisodrRb2uTGvrFsg?e=5j1nx4).

To link to checkpoint when testing or predicting, override/set path under `checkpoint_file:<path>` in `conf/main_config.yaml.` e.g.

`
python test.py dset=<dset-name> experiment=<experiment-name> +checkpoint_file=<path to checkpoint.th file>
`

Alternatively, make sure that the checkpoint file is in its corresponding output folder:  
For each low to high resolution setting, hydra creates a folder under `outputs/<dset-name>/<experiment-name>`

Make sure that `restart: false` in `conf/main_config.yaml`

### Citation 

@inproceedings{Abreu2024lamir,
        author    = {Wallace Abreu and Luiz Wagner Pereira Biscainho},
        title     = {AEROMamba: An Efficient Architecture for Audio Super-Resolution Using Generative Adversarial Networks and State Space Models},
        booktitle = {Proceedings of the 1st Latin American Music Information Retrieval Workshop},
        year      = {2024},
        address   = {Rio de Janeiro, Brazil},
      }
      
