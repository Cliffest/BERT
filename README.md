# BERT on 《倚天屠龙记》

## Results
<div style="display:flex; justify-content:space-between;">
  <img src="Results/result_at_epoch_0.png" alt="Image 1" style="width:250px;">
  <img src="Results/result_at_epoch_100.png" alt="Image 2" style="width:250px;">
  <img src="Results/result_at_epoch_540.png" alt="Image 3" style="width:250px;">
</div>

## Download
Only codes.
```
git clone https://github.com/Cliffest/BERT.git
```
Codes and model weights (require Git LFS).
```
git lfs clone https://github.com/Cliffest/BERT.git
```

## Requirements
Require GPU.
```bash
# conda env export > environment.yml
conda env update -n bert -f environment.yml
conda activate bert
```

## Train
```bash
python train.py --data_path data/倚天屠龙记_train_no-space.txt --output_dir outputs --batch_size 32
python train.py --data_path data/倚天屠龙记_train_no-space.txt --output_dir outputs --batch_size 32 --epochs 1000 --resume_from_epoch 9 --save_interval 10
```
Save model at epoch $N$ (count from 0).
```bash
python train.py --data_path data/倚天屠龙记_train_no-space.txt --output_dir outputs --epochs N+1 --resume_from_epoch N
```

## Test
```bash
# python train.py --data_path data/倚天屠龙记_train_no-space.txt --output_dir outputs --epochs 101 --resume_from_epoch 100
python test.py --model_dir outputs --n_epoch 100 --mask_token_ids 3
```

For example, commands to run the model that has been trained for 540 epochs.
```bash
# Get model files from checkpoint_epoch_540.pth
python train.py --data_path data/倚天屠龙记_train_no-space.txt --output_dir outputs --epochs 541 --resume_from_epoch 540
# Test instance
python test.py --model_dir outputs --n_epoch 540 --mask_token_ids 3 28
```