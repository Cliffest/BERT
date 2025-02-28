# BERT on 《倚天屠龙记》

## Requirements
```
conda create -n bert python=3.11
conda activate bert

# cpu
pip install torch torchvision torchaudio
# gpu-CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers
pip install scikit-learn
```