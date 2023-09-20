#!/usr/local_rwth/bin/zsh

# run training

python3 -W ignore tag_program  \
--global_batch_size=tag_batch \
--augment=tag_aug \
--epoch=tag_epoch 2>&1
