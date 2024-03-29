#!/bin/bash

[ -f path.sh ] && . ./path.sh

if [ $# != 3 ] ; then 
  echo "usage: test.sh <CUDA_DEVICE> <num of all_utts> <num of paired> <unit_type>"
  echo "e.g. test.sh 0 -1 -1 char"
  exit 1
fi

#RNN_module=$1
device_id=$1
num_all_utts=$2
num_paired=$3
unit_type=$4

init_lr=0.0001
batch_size=16
seq_len=165
feat_dim=39
hidden_dim=256
enc_num_layers=1
dec_num_layers=1
dropout_rate=0.3
D_num_layers=3
iter_d=3
neg_num=1

weight_in_r=0.2
weight_cr_r=0.
weight_in_txt_ce=1.
weight_cr_txt_ce=1.
weight_x=0.
#weight_pos_spk=1.
#weight_neg_spk=1.
weight_pos_paired=5.
weight_neg_paired=5.
#pos_thres=0.01
neg_num=1

top_NN=100
width=30
weight_LM=0.01

model_dir=$exp_dir/model_${unit_type}_${init_lr}_${num_all_utts}_${num_paired}_${hidden_dim}_${enc_num_layers}_${dec_num_layers}_${dropout_rate}_${weight_in_r}_${weight_cr_r}_${weight_in_txt_ce}_${weight_cr_txt_ce}_${weight_x}_${weight_pos_paired}_${weight_neg_paired}_${neg_num}
log_dir=$exp_dir/log_${unit_type}_${init_lr}_${num_all_utts}_${num_paired}_${hidden_dim}_${enc_num_layers}_${dec_num_layers}_${dropout_rate}_${weight_in_r}_${weight_cr_r}_${weight_in_txt_ce}_${weight_cr_txt_ce}_${weight_x}_${weight_pos_paired}_${weight_neg_paired}_${neg_num}
result_dir=$exp_dir/result_${unit_type}_${init_lr}_${num_all_utts}_${num_paired}_${hidden_dim}_${enc_num_layers}_${dec_num_layers}_${dropout_rate}_${weight_in_r}_${weight_cr_r}_${weight_in_txt_ce}_${weight_cr_txt_ce}_${weight_x}_${weight_pos_paired}_${weight_neg_paired}_${neg_num}

train_meta_pkl=$feat_dir/processed/timit-train-meta.pkl       # {'prefix': [4620 x drID_spkID_uttID]}
train_mfcc_pkl=$feat_dir/processed/timit-train-mfcc-nor.pkl   # [4620 x num_of_frames x 39]
train_phn_pkl=$feat_dir/processed/timit-train-phn-39.pkl      # [4620 x num_of_phns x [phn, start, end]] ** include 'h#' **
train_wrd_pkl=$feat_dir/processed/timit-train-wrd.pkl         # [4620 x num_of_wrds x [wrd, start, end]] 
train_slb_pkl=$feat_dir/processed/timit-train-slb.pkl         # [4620 x num_of_slbs x [slb, start, end]] 

test_meta_pkl=$feat_dir/processed/timit-test-meta.pkl         # {'prefix': [? x drID_spkID_uttID]}
test_mfcc_pkl=$feat_dir/processed/timit-test-mfcc-nor.pkl     # [? x num_of_frames x 39]
test_phn_pkl=$feat_dir/processed/timit-test-phn-39.pkl        # [? x num_of_phns x [phn, start, end]] ** include 'h#' **
test_wrd_pkl=$feat_dir/processed/timit-test-wrd.pkl           # [? x num_of_wrds x [wrd, start, end]] 
test_slb_pkl=$feat_dir/processed/timit-test-slb.pkl           # [? x num_of_slbs x [slb, start, end]] 

### training ###
export CUDA_VISIBLE_DEVICES=$device_id

#if [ "$RNN_module" != "default" ]; then
  #$py_file=$path/RNN/train_RNN.py
#else
  #$py_file=$path/ONLSTM/train_ONLSTM.py
#fi

mode=test
n_epochs=0

python3 $path/main.py --init_lr=$init_lr --batch_size=$batch_size --seq_len=$seq_len --feat_dim=$feat_dim \
  --p_hidden_dim=$hidden_dim --phn_num_layers=$enc_num_layers --dec_num_layers=$dec_num_layers --dropout_rate=$dropout_rate --neg_num=$neg_num \
  --weight_in_r=$weight_in_r --weight_cr_r=$weight_cr_r --weight_in_txt_ce=$weight_in_txt_ce --weight_cr_txt_ce=$weight_cr_txt_ce \
  --weight_x=$weight_x --weight_pos_paired=$weight_pos_paired --weight_neg_paired=$weight_neg_paired \
  --top_NN=$top_NN --width=$width --weight_LM=$weight_LM --n_epochs=$n_epochs \
  $train_meta_pkl $train_mfcc_pkl $train_phn_pkl $train_wrd_pkl $train_slb_pkl \
  $test_meta_pkl $test_mfcc_pkl $test_phn_pkl $test_wrd_pkl $test_slb_pkl \
  $log_dir $model_dir $result_dir $mode -1 -1 $unit_type
