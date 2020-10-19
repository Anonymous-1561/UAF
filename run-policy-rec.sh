nohup python policy_rec.py \
--name coldrec3-conv-hard-c128 \
--type conv --method hard \
--data_file rec/coldrec3/data_finetune.csv --log_every 50 \
--pre store/pre-pre/coldrec3-pre-10.19-15.02.53 \
--temp 10 \
--n_neg 99 \
--lr 0.0001 \
--batch_size 128 \
--channel 128 \
--kernel_size 3 \
--n_blocks 4 --block_shape 1,4 \
--store_root store/policy-rec \
--iter 50 \
--gpu 5 \
> output.log &

# choose policy type and forward method
# --type conv --method hard \
# --type conv --method soft \
# --type gru --method hard \
# --type gru --method soft \
# --type random