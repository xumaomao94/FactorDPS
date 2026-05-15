# FactorDPS

This repository contains the demo code for our paper:

**"Completing Radio Map Tensors via Factor Diffusion Posterior Sampling"**

## Instructions

To run the demo code, please follow the steps below.

### 1. Set up the Python environment

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

### 2. Download the trained models

Download the trained models from the links below and place them in the `./models/` directory.

- **SLF model:** [Download SLF model](https://drive.google.com/file/d/1KBf7PjR4_JLP3Y8dpcaTUNexqhEyr4Yx/view?usp=drive_link)
- **PSD model:** [Download PSD model](https://drive.google.com/file/d/1IPlyERTTcFSU5a8N53_s_KlgM4iGL7xv/view?usp=drive_link)

The directory structure should look like:

```text
FactorDPS/
├── models/
│   ├── <SLF trained model files>
│   └── <PSD trained model files>
```

### 3. Download the dataset

Download the RadioMapSeer dataset from:

[https://radiomapseer.github.io](https://radiomapseer.github.io)

After downloading the dataset, modify the `data_root_dir` variable in the demo scripts to point to your local dataset directory.

---

## File Descriptions

### Training

To train the models, run the corresponding shell scripts:

```bash
bash train_xxx.sh
```

The main training-related files are:

- `main_train_xxx.py`  
  Main training script called by the corresponding `.sh` files. It defines the training procedure.

- `config/SLF_RadioMapSeer_DDPM_conditional_625Buildings.json`  
  Configuration file for SLF diffusion model training, including the training settings, dataset configuration, and network architecture.

- `config/PSD_DDPM.json`  
  Configuration file for PSD diffusion model training, including the training settings, dataset configuration, and network architecture.

### Demo and Inference

- `demo_xxx.py`  
  Demo scripts for different experimental settings, including different numbers of frequency bands, different numbers of emitters, and different measurement degradation models such as central missing observations and quantized observations.

- `run_method.py`  
  Implementation of the DPS-based inference algorithm used by the demo scripts. This file contains the proposed FactorDPS algorithm.

- `inference.json`  
  Configuration file for inference, including the paths to the trained models.


## Code Credits

The initial implementation of this codebase was developed by Rajesh Shrestha. The code was later finetuned and organized by Le Xu.
