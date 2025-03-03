# BERT on 《倚天屠龙记》

## Requirements
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
python test.py --model_dir outputs --n_epoch 100 --mask_token_ids 3
```