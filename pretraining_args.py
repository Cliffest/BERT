# -----------ARGS---------------------
pretrain_train_path = "data/倚天屠龙记_train.txt"  # Require: One sentence per line.
pretrain_dev_path = "data/倚天屠龙记_dev.txt"  # Require: One sentence per line.

bert_config_json = "./bert_config.json"
vocab_file = "data/bert_vocab.txt"
output_dir = "outputs"

no_cuda = False

max_seq_length = 256
do_train = True
do_lower_case = True
train_batch_size = 32
eval_batch_size = 32
learning_rate = 1e-4
num_train_epochs = 6
warmup_proportion = 0.1
local_rank = -1
seed = 42
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
masked_lm_prob = 0.15
max_predictions_per_seq = 20