#!/bin/bash

NAME=$1
GPU=$2
DATASET=$3
LOG_DIR=logs/$DATASET/$NAME
LOG_FILE=$LOG_DIR/train.log
mkdir -p $LOG_DIR

mkdir -p $LOG_DIR/python
cp -f *.py $LOG_DIR/python/
mkdir -p $LOG_DIR/python/data
cp -f ./data/*.py $LOG_DIR/python/data/
mkdir -p $LOG_DIR/python/utils
cp -f ./utils/*.py $LOG_DIR/python/utils/
mkdir -p $LOG_DIR/python/options
cp -f ./options/*.py $LOG_DIR/python/options/
mkdir -p $LOG_DIR/python/motion_loaders
cp -f ./motion_loaders/*.py $LOG_DIR/python/motion_loaders/
mkdir -p $LOG_DIR/python/models
cp -f ./models/*.py $LOG_DIR/python/models/
mkdir -p $LOG_DIR/python/models/vq
cp -f ./models/vq/*.py $LOG_DIR/python/models/vq
mkdir -p $LOG_DIR/python/models/transformer
cp -f ./models/transformer/*.py $LOG_DIR/python/models/transformer

echo NAME $1 | tee $LOG_FILE
echo GPU $2 | tee $LOG_FILE -a
echo DATASET $3 | tee $LOG_FILE -a
echo LOG_DIR $LOG_DIR | tee $LOG_FILE -a
echo LOG_FILE $LOG_FILE | tee $LOG_FILE -a
echo EXTRA ${@:3} | tee $LOG_FILE -a


python train_vq.py --name $NAME --gpu_id $GPU --dataset_name $DATASET     \
    ${@:4} | tee $LOG_DIR/train.log -a


# bash run_rvq.sh rvq_test 0 humanml3d --batch_size 256 --num_quantizers 6  --max_epoch 50 --quantize_dropout_prob 0.2 --gamma 0.1 --code_dim2d 1024 --nb_code2d 256