nohup python pretrain.py \
--name coldrec3-pre \
--data_file rec/coldrec3/data_pretrain.csv \
--batch_size 128 \
--log_every 50 \
--channel 128 \
--kernel_size 3 \
--store_root store/pre-pre \
--lr 0.0001 \
--n_blocks 4 --block_shape 1,4 \
--iter 50 \
--gpu 5  \
> output.log &

# --occupy 0.95