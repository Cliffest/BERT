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
python train.py --data_path data/倚天屠龙记_train_no-space.txt --output_dir outputs --batch_size 32 --epochs 100 --resume_from_epoch 9
```

## Test
```bash
python test.py --n_epoch 100 --mask_token_ids 3
```