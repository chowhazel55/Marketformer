# Marketformer

This repository contains the official implementation of the paper:  
**Marketformer: Virtual Market Transformer for Stock Price Forecasting**

---

## Environment Setup

```bash
conda create -n Marketformer python=3.10.16 -y
conda activate Marketformer

# Install PyTorch with CUDA (replace cuXXX with your CUDA build, e.g., cu118 / cu121)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX

# Install other dependencies
pip install -r requirements.txt


## Training

### Option 1: Run the bash script
```bash
cd scripts
sh bash.sh
### Option 1: Run the bash script
```bash
cd scripts
python train.py -d DATASET_NAME -m MODEL_NAME -g GPU_ID -l LEARNING_RATE --seed RANDOM_SEED

