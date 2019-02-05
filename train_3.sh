#!/bin/bash

[ -f path.sh ] && . ./path.sh

if [ $# != 3 ] ; then 
  echo "usage: train.sh <CUDA_DEVICE> <num of paired> <n_epochs for training>"
  echo "e.g. train.sh 0 1000 300"
  exit 1
fi

#RNN_module=$1
device_id=$1
num_paired=$2
n_epochs=$3

init_lr=0.0005
batch_size=32
seq_len=165
feat_dim=39
hidden_dim=256
enc_num_layers=2
dec_num_layers=2
D_num_layers=3
dropout_rate=0.1
iter_d=3
#chunk_size=1
#pred_step=12
#pred_neg_num=8
#AR_scale=1.
#TAR_scale=1.

mkdir -p $exp_dir

model_dir=$exp_dir/model_${init_lr}_${hidden_dim}_${enc_num_layers}_${dec_num_layers}_${D_num_layers}_${dropout_rate}_${iter_d}
log_dir=$exp_dir/log_${init_lr}_${hidden_dim}_${enc_num_layers}_${dec_num_layers}_${D_num_layers}_${dropout_rate}_${iter_d}
result_dir=$exp_dir/result_${init_lr}_${hidden_dim}_${enc_num_layers}_${dec_num_layers}_${D_num_layers}_${dropout_rate}_${iter_d}

mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $result_dir

train_meta_pkl=$feat_dir/processed/timit-train-meta.pkl       # {'prefix': [4620 x drID_spkID_uttID]}
train_mfcc_pkl=$feat_dir/processed/timit-train-mfcc-nor.pkl   # [4620 x num_of_frames x 39]
train_phn_pkl=$feat_dir/processed/timit-train-phn.pkl      # [4620 x num_of_phns x [phn, start, end]] ** include 'h#' **
train_wrd_pkl=$feat_dir/processed/timit-train-wrd.pkl         # [4620 x num_of_wrds x [wrd, start, end]] 
train_slb_pkl=$feat_dir/processed/timit-train-slb.pkl         # [4620 x num_of_slbs x [slb, start, end]] 

test_meta_pkl=$feat_dir/processed/timit-test-meta.pkl         # {'prefix': [? x drID_spkID_uttID]}
test_mfcc_pkl=$feat_dir/processed/timit-test-mfcc-nor.pkl     # [? x num_of_frames x 39]
test_phn_pkl=$feat_dir/processed/timit-test-phn.pkl        # [? x num_of_phns x [phn, start, end]] ** include 'h#' **
test_wrd_pkl=$feat_dir/processed/timit-test-wrd.pkl           # [? x num_of_wrds x [wrd, start, end]] 
test_slb_pkl=$feat_dir/processed/timit-test-slb.pkl           # [? x num_of_slbs x [slb, start, end]] 

### training ###
export CUDA_VISIBLE_DEVICES=$device_id

#if [ "$RNN_module" != "default" ]; then
  #$py_file=$path/RNN/train_RNN.py
#else
  #$py_file=$path/ONLSTM/train_ONLSTM.py
#fi

mode=train

python3 $path/main.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim --p_hidden_dim=$hidden_dim \
  --phn_num_layers=$enc_num_layers --dec_num_layers=$dec_num_layers --D_num_layers=$D_num_layers --dropout_rate=$dropout_rate \
  --iter_d=$iter_d --n_epochs=$n_epochs \
  $train_meta_pkl $train_mfcc_pkl $train_phn_pkl $train_wrd_pkl $train_slb_pkl \
  $test_meta_pkl $test_mfcc_pkl $test_phn_pkl $test_wrd_pkl $test_slb_pkl \
  $log_dir $model_dir $result_dir $mode $num_paired
