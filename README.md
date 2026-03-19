# Kinyarwanda XTTS Fine-Tuning Pipeline

This folder contains the complete, ready-to-run pipeline to fine-tune Coqui XTTS v2 on your private HuggingFace dataset (`Washere-1/tecgrw-audio`).

## Prerequisites

1.  **Hardware:** You MUST run this on a machine with a dedicated Nvidia GPU (RTX 3090/4090 or an A100/A10G). You can rent a cloud VM on RunPod or Google Cloud if your local machine does not have a 24GB VRAM GPU.
2.  **OS:** Linux (Ubuntu 22.04+) is strongly recommended for Coqui TTS compilation, though Windows with WSL2 will work.

## Run Instructions

### Step 1: Install Python Environment
Open a terminal in this folder and run:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install setuptools wheel numpy==1.24.3
pip install -r requirements.txt
```

### Step 2: Download the Base Model
XTTS fine-tuning requires the base, multi-lingual model to start from.
1. Go to HuggingFace: https://huggingface.co/coqui/XTTS-v2/tree/main 
2. Create a folder in this directory called `base_xtts_v2`
3. Download `model.pth`, `config.json`, and `vocab.json` into that folder.

### Step 3: Format the HuggingFace Dataset
We need to convert the dataset into the standard `LJSpeech` array format containing 22050Hz WAV files.
1. Open `prepare_dataset.py` and put your HuggingFace Token in `HF_TOKEN = "your_token"` (required because your dataset is Private).
2. Run the script:
```bash
python prepare_dataset.py
```
This will create a `kinyarwanda_tts_dataset` folder with all your audio and a formatted `metadata.csv`.

### Step 4: Start Training!
Once the dataset is ready, ignite the PyTorch trainer!
```bash
python train_xtts.py
```
This will run for 10-24 hours depending on your GPU. 

### Step 5: Deployment
Once complete, you will find a new `model.pth` and `config.json` inside the `xtts_ft_output` directory. Copy those two files into your Tekana production `tts_service` container's `model` folder, and it will immediately start speaking with a native Kinyarwanda accent!
