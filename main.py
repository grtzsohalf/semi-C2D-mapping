# -*- coding: utf-8 -*-
"""
Semi-Continuous-to-discrete Domain Mapping (semi-C2D-mapping)
*************************************************************
**Author**: `Yi-Chen Chen <https://github.com/grtzsohalf/semi-C2D-mapping>`_

"""

import os
import sys
import argparse
from solver import Solver
from process_data import Speech
from saver import PytorchSaver 
from utils import count_LM


######################################################################
# Training
# ========
#


######################################################################
# Parser
# ------
#

def add_parser():
    parser = argparse.ArgumentParser(prog="PROG", 
        description='Audio2vec Training Script')
    parser.add_argument('--init_lr',  type=float, default=0.0005,
        metavar='<--initial learning rate>')
    parser.add_argument('--batch_size',type=int, default=64,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--seq_len',type=int, default=50,
        metavar='--<seq len>',
        help='length of a sequence')
    parser.add_argument('--feat_dim',type=int, default=39,
        metavar='--<feat dim>',
        help='feature dimension')
    parser.add_argument('--p_hidden_dim',type=int, default=256,
        metavar='<--phonetic encoder hidden dimension>',
        help='The hidden dimension of a neuron in phonetic encoder')
    # parser.add_argument('--s_hidden_dim',type=int, default=256,
        # metavar='<--speaker encoder hidden dimension>',
        # help='The hidden dimension of a neuron in speaker encoder')
    parser.add_argument('--phn_num_layers',type=int, default=3,
        metavar='--<number of phonetic encoder rnn layers>',
        help='number of phonetic encoder rnn layers')
    # parser.add_argument('--spk_num_layers',type=int, default=3,
        # metavar='--<number of speaker encoder rnn layers>',
        # help='number of speaker encoder rnn layers')
    parser.add_argument('--dec_num_layers',type=int, default=3,
        metavar='--<number of decoder rnn layers>',
        help='number of decoder rnn layers')
    # parser.add_argument('--D_num_layers',type=int, default=3,
        # metavar='--<number of discriminator fully-connected layers>',
        # help='number of phonetic discriminator fully-connected layers')
    parser.add_argument('--dropout_rate',type=float, default=0.3,
        metavar='--<dropout rate of encoder/decoder>',
        help='dropout rate of encoder/decoder')
    # parser.add_argument('--iter_d',type=int, default=3,
        # metavar='--<num_iterations of D while updating G once>',
        # help='num_iterations of D while updating G once')

    parser.add_argument('--weight_r',type=float, default=1.,
        metavar='--<weight of r_loss>',
        help='weight of r_loss')
    parser.add_argument('--weight_txt_r',type=float, default=1.,
        metavar='--<weight of txt_r_loss>',
        help='weight of txt_r_loss')
    # parser.add_argument('--weight_g',type=float, default=1.,
        # metavar='--<weight of g_loss>',
        # help='weight of g_loss')
    parser.add_argument('--weight_pos_spk',type=float, default=1.,
        metavar='--<weight of pos_spk_loss>',
        help='weight of pos_spk_loss')
    parser.add_argument('--weight_neg_spk',type=float, default=1.,
        metavar='--<weight of neg_spk_loss>',
        help='weight of neg_spk_loss')
    parser.add_argument('--weight_pos_paired',type=float, default=1.,
        metavar='--<weight of pos_paired_loss>',
        help='weight of pos_paired_loss')
    parser.add_argument('--weight_neg_paired',type=float, default=1.,
        metavar='--<weight of neg_paired_loss>',
        help='weight of neg_paired_loss')
    # parser.add_argument('--weight_d',type=float, default=1.,
        # metavar='--<weight of d_loss>',
        # help='weight of d_loss')
    # parser.add_argument('--weight_gp',type=float, default=1.,
        # metavar='--<weight of gp_loss>',
        # help='weight of gp_loss')

    parser.add_argument('--width',type=int, default=10,
        metavar='--<beam width>',
        help='beam width')
    parser.add_argument('--weight_LM',type=float, default=1.,
        metavar='--<weight of LM scores>',
        help='weight of LM_scores')

    parser.add_argument('--n_epochs',type=int, default=20,
        metavar='--<# of epochs for training>',
        help='The number of epochs for training')

    parser.add_argument('train_meta', 
        metavar='<training metadata>')    
    parser.add_argument('train_mfcc', 
        metavar='<training mfcc features>')    
    parser.add_argument('train_phn', 
        metavar='<training phonemes with start/end time>')    
    parser.add_argument('train_wrd', 
        metavar='<training words with start/end time>')    
    parser.add_argument('train_slb', 
        metavar='<training syllables with start/end time>')    

    parser.add_argument('test_meta', 
        metavar='<testing metadata>')    
    parser.add_argument('test_mfcc', 
        metavar='<testing mfcc features>')    
    parser.add_argument('test_phn', 
        metavar='<testing phonemes with start/end time>')    
    parser.add_argument('test_wrd', 
        metavar='<testing words with start/end time>')    
    parser.add_argument('test_slb', 
        metavar='<testing syllables with start/end time>')    

    parser.add_argument('log_dir', 
        metavar='<log directory>')
    parser.add_argument('model_dir', 
        metavar='<model directory>')
    parser.add_argument('result_dir', 
        metavar='<result directory>')
    parser.add_argument('mode', 
        metavar='<mode (train or test)>')    
    parser.add_argument('num_paired', type=int, default=-1,
        metavar='<number of paired data>')    

    return parser


######################################################################
# Main function
# -------------
#

if __name__ == '__main__': 

    parser = add_parser()
    FLAG = parser.parse_args()


    ######################################################################
    # Load and process data
    #
    
    if FLAG.mode == 'train':
        train_data = Speech('train', FLAG.batch_size, FLAG.num_paired)
        train_data.process_data(FLAG.train_meta, FLAG.train_mfcc, FLAG.train_phn, FLAG.train_wrd, FLAG.train_slb)
        test_data = Speech('test', FLAG.batch_size, FLAG.num_paired)
        test_data.process_data(FLAG.test_meta, FLAG.test_mfcc, FLAG.test_phn, FLAG.test_wrd, FLAG.test_slb)
    else:
        test_data = Speech('test', FLAG.batch_size, FLAG.num_paired)
        test_data.process_data(FLAG.test_meta, FLAG.test_mfcc, FLAG.test_phn, FLAG.test_wrd, FLAG.test_slb)
        train_data = Speech('train', FLAG.batch_size, FLAG.num_paired)
        train_data.process_data(FLAG.train_meta, FLAG.train_mfcc, FLAG.train_phn, FLAG.train_wrd, FLAG.train_slb)
    print ("Data processed!")

    # count LM
    wrds = []
    for wrd_meta_utt in train_data.wrd_meta:
        wrds.append([w[0] for w in wrd_meta_utt])
    for wrd_meta_utt in test_data.wrd_meta:
        wrds.append([w[0] for w in wrd_meta_utt])
    LM = count_LM(wrds) 
    train_data.LM = LM
    test_data.LM = LM
    # sorted_LM = sorted(LM.items(), key=lambda kv: kv[1], reverse=True)
    # with open('TIMIT_utils/LM.txt', 'w') as f:
        # for pair, log_prob in sorted_LM:
            # f.write(pair[0]+' '+pair[1]+' '+pair[2]+': '+str(log_prob)+'\n')
    # exit()


    ######################################################################
    # Construct a Solver
    #

    solver = Solver(FLAG.init_lr, FLAG.batch_size, FLAG.seq_len, FLAG.feat_dim, FLAG.p_hidden_dim,
                    FLAG.phn_num_layers, FLAG.dec_num_layers, FLAG.dropout_rate, 
                    FLAG.weight_r, FLAG.weight_txt_r, FLAG.weight_pos_spk,
                    FLAG.weight_neg_spk, FLAG.weight_pos_paired, FLAG.weight_neg_paired, 
                    FLAG.width, FLAG.weight_LM, FLAG.log_dir, FLAG.mode)
    solver.build_model()
    print ("Solver constructed!")


    ######################################################################
    # Check if checkpoints exist
    #

    save_dir = os.path.join(FLAG.model_dir)
    # resume_dir = os.path.join(FLAG.model_dir, 'resume')
    if FLAG.mode == 'train':
        saver = PytorchSaver(100, save_dir)
        global_step = 0
        if os.listdir(save_dir):
            print ('Loading model...')
            print (save_dir)
            model_name, state_dict = PytorchSaver.load_dir(save_dir)
            print ('Loaded model: '+model_name)
            # for k in state_dict['state_dict']:
                # print (k)
            # print ('\n')
            # for name, param in solver.model.named_parameters():
                # if param.requires_grad:
                    # print (name)
            solver.model.load_state_dict(state_dict['state_dict'])
            global_step = int(model_name.split('_')[1])
    else:
        if not os.path.exists(save_dir):
            print("save_dir should exist in eval mode", file=sys.stderr)
            sys.exit(-1)
        print ('Loading model...')
        model_name, state_dict = PytorchSaver.load_dir(save_dir)
        print ('Loaded model: '+model_name)
        solver.model.load_state_dict(state_dict['state_dict'])


    ######################################################################
    # Training and Evaluating
    #

    if FLAG.mode == 'train':
        print ('Start training!')
        solver.train_iters(train_data, test_data, saver, FLAG.n_epochs, global_step, FLAG.result_dir)
    else:
        print ('Start testing!')
        solver.evaluate(test_data, train_data, FLAG.result_dir)
