#!/usr/local_rwth/bin/zsh

# run training

python3 -W ignore tag_program  \
--global_batch_size=tag_batch \
--augment=tag_aug \
--image_dir=${WORK}/repos/distributed-ml/pytorch/data/images_collective \
--mask_dir=${WORK}/repos/distributed-ml/pytorch/data/masks_collective \
--epoch=tag_epoch 2>&1
