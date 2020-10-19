nohup python policy_rl_rec.py \
--name coldrec3-rl-conv \
--type conv \
--data_file rec/coldrec3/data_finetune.csv --log_every 50 \
--pre store/pre-pre/coldrec3-pre-10.19-15.02.53 \
--temp 10 \
--n_neg 99 \
--lr 0.0001 \
--batch_size 128 \
--channel 128 \
--kernel_size 3 \
--n_blocks 4 --block_shape 1,4 \
--store_root store/policy-rl-rec \
--iter 50 --rl_start_iter 5 \
--gpu 1 \
--gamma 1.0 --reward_k 5 \
> output.log &

# choose policy type
# --type conv \
# --type gru \
