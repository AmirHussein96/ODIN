# -*- coding: utf-8 -*-
"""
Copyright (C) 2020 American University of Beirut
                        Amir Hussein
                        
Main file to run the ODIN model
Three adaptation scenarios:
    Cross user
    Cross device
    Cross user cross device

"""
#import necessary libraries

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import sys
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from utils import gen_noise, save_file, compare_recon
from models import mmd_loss, fixprob, Source_model, ODIN
from data_preparation import windowing, extract_users, down_sampling2, load_data
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import pdb
import datetime
import logging
import copy
#tf.disable_v2_behavior()
sns.set(style='whitegrid', palette='deep', font_scale=1.5)

#import random
#random.seed(5000)
#np.random.seed(5000)
#tf.set_random_seed(5000)

if not sys.warnoptions:
    warnings.simplefilter("ignore")



time_now = datetime.datetime.now()
# cross user cross device


def prep_data(data_t, data_s):
    """
    Normalizaing and splitting the data for training and adaptation
    """        
    X_t,Y_t = windowing(data_t) 
    X_s,Y_s = windowing(data_s)  

    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split( X_s, Y_s, test_size=0.1, stratify=Y_s)
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_t, Y_t, test_size=0.1, stratify=Y_t)
    
    norm = StandardScaler()
    norm.fit(np.vstack([X_train_s,X_train_t]).reshape(-1,3))
    X_train_s = norm.transform(X_train_s.reshape(-1,3)).reshape(-1,128,3)
    X_train_t = norm.transform(X_train_t.reshape(-1,3)).reshape(-1,128,3)
    # X_train_s_n = norm.transform(X_train_s_n.reshape(-1,3)).reshape(-1,128,3)
    # X_train_t_n = norm.transform(X_train_t_n.reshape(-1,3)).reshape(-1,128,3)
    X_train_s_n = X_train_s + gen_noise(X_train_s.shape, X_train_s, False)
    X_train_t_n = X_train_t + gen_noise(X_train_t.shape, X_train_t, False)
    X_val_s = norm.transform(X_val_s.reshape(-1,3)).reshape(-1,128,3)
    X_val_t = norm.transform(X_val_t.reshape(-1,3)).reshape(-1,128,3)

    return X_train_s, X_train_s_n, y_train_s, X_val_s, y_val_s, X_val_t, y_val_t, X_t, X_train_t_n, Y_t, norm

def get_args():
    # Get some basic command line arguements
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', 
                        help='Training mode (cr_user, cr_device, cr_user_device)', 
                        type=str, default='cr_user_device')
    parser.add_argument('-u', '--user', 
                        help='Target participant', 
                        type=str, default='g')
    parser.add_argument('-s', '--steps', help='Pretraining steps for source model', 
                        type=int, default=6000)
    parser.add_argument('-b', '--batch', help='Training batch size', 
                        type=int, default=256)
    parser.add_argument('-d', '--dataset', help='Dataset [HAR|PAR]', 
                        type=str, default='HAR')
    parser.add_argument('-c', '--device', help='Wearable device [watch|phone]', 
                        type=str, default='phone')
    parser.add_argument('-l', '--da_loss', help='Adaptation objective [MMD, DC]', 
                        type=str, default='MMD')
    #parser.add_argument('-p', '--path', help='Dataset path', 
     #                   type = str, default='datasets/')
    
    return parser.parse_args()


def train(X_train_s, X_train_s_n, y_train_s, X_val_s, y_val_s, 
                                X_val_t, y_val_t, X_t, X_t_n, Y_t, norm, args, source_target=None):
    
    logging.basicConfig(handlers=[logging.FileHandler(filename="./log_{}_flops.txt".format(str(args.mode)), encoding='utf-8', mode='a+')],format="%(asctime)s :%(levelname)s: %(message)s", datefmt="%F %A %T", level=logging.INFO)
    
    n_lables = y_train_s.shape[1]
    batch_size = args.batch
    data = [X_t, X_t_n, Y_t, X_train_s, X_train_s_n, y_train_s, X_val_t, y_val_t, X_val_s, y_val_s]
    
    
    if args.mode =='cr_user':
            save_path = os.path.join("results",args.mode,args.dataset+ str(args.da_loss), args.device + "_" +str(time_now.strftime("%y_%m_%d_%H")))
            source = args.user
    elif args.mode =='cr_device' :
            save_path = os.path.join("results",args.mode,args.dataset + str(args.da_loss), source_target[0]+"_"+str(time_now.strftime("%y_%m_%d_%H")))
            source = source_target[1]
    else:
            save_path = os.path.join("results",args.mode,args.dataset + str(args.da_loss), args.user+"_"+str(time_now.strftime("%y_%m_%d")))
            source = source_target[1]
    path_result = os.path.join(str(save_path),'f1')
    if not os.path.isdir(path_result):
            os.makedirs(path_result, exist_ok=True)
    
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    
    # adaptation with odin 
    
    source_model = Source_model(batch_size, args, n_lables)
    print('\n Pretraining only source model \n')
    #X_train_s.shape[0]*50//128
    source_only_emb,sess,hist = \
        source_model.train_and_evaluate(graph, data, logging, 
                                        num_steps=6000, 
                                        verbose=True)
    feature, label_class, decode_w = source_model.parameters()
    pretrained_par = [feature, label_class, decode_w]
    # save model weights
    weights_path = os.path.join(str(save_path),"weights")
    if not os.path.isdir(weights_path):
        os.makedirs(weights_path, exist_ok=True)
    save_file(feature, os.path.join(str(weights_path),'%s_%s.hkl'%('feat_ext',source)))
    save_file(label_class, os.path.join(str(weights_path), '%s_%s.hkl'%('label_class',source)))
    save_file(decode_w, os.path.join(str(weights_path),'%s_%s.hkl'%('decode',source)))
    
    source_model.sess.close()
        
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    run_meta = tf.RunMetadata()
        
    with graph.as_default():  
        X_t = np.concatenate((X_t, X_val_t),axis=0)
        Y_t = np.concatenate((Y_t, y_val_t),axis=0)
        odin = ODIN(args.da_loss, pretrained_par, n_lables, args, batch_size*2)
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(graph, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            print('Flops: ',flops.total_float_ops)
            logging.info('Flops: %s'%(str(flops.total_float_ops)))
        #X_t.shape[0]*6//128
        print('\n Adaptation to target %s'+ source_target[1])
        test_emb, sess, history, combined_test_labels, combined_test_domain, X0 = \
                odin.train_and_evaluate(data, graph, norm, logging,
                                        args.user,num_steps=1000)
        
    save_file(history, os.path.join(str(path_result),'%s.hkl'%(source)))
       
    #compare_recon(sess, X_val_t, odin, 0, False) # plotting reconstructed signals
    logging.shutdown()
        
def main():
    
    args = get_args()
    if args.dataset == "PAR":
        args.path = "C:/Users/anh21/OneDrive - American University of Beirut/Advanced_DANN_project"
    else:
       args.path = "datasets"
    if args.mode == 'cr_user_device':
        # Loading the data
        df = pd.read_csv(os.path.join(args.path, "Phones_accelerometer.csv"))
        df = df.dropna()
        df.rename(columns={'x': 'attr_x', 'y': 'attr_y','z': 'attr_z','gt':'activity'}, inplace=True)
        #device_ss = ['nexus4','s3','samsungold','s3mini']
        device_ss = ['samsungold']       # SamsungS+
        device_tt = {'s3mini':100,'s3':150,'nexus4':200}
        #device_tt={'s3mini':100}
        user_tt = ['f','d','e']                       # change the user target
        for user_t in user_tt:
            for device_s in device_ss: 
                args.user = user_t
               # all_usr = ['a','b','g','c','h']
                #all_usr.remove(user_t)
                data_s = extract_users(df,['a','b','g','c','h'])
                data_s = data_s[data_s['Model'] == device_s]
                for device in device_tt:
                    
                    if device != device_s:
                        data_t = extract_users(df,[user_t])
                        data_t = data_t[data_t['Model'] == device]
                        
                        # downsampling 
                        if device != 'samsungold' and device_s !='samsungold' :
                             data_t = down_sampling2(data_t,device_tt[device], 50)
                             data_s = down_sampling2(data_s,device_tt[device_s], 50)
                            
                        elif device != 'samsungold' and device_s =='samsungold':
                             
                             data_t = down_sampling2(data_t,device_tt[device], 50)
                             
                        elif device == 'samsungold' and device_s !='samsungold':
                            data_s = down_sampling2(data_s,device_tt[device_s], 50)
                        X_train_s, X_train_s_n, y_train_s, X_val_s, \
                            y_val_s, X_val_t, y_val_t, X_t, \
                              X_t_n, Y_t, norm = prep_data(data_t, data_s)
                                
                       # training and adapting
                       
                        train(X_train_s, X_train_s_n, y_train_s, X_val_s, y_val_s, 
                                    X_val_t, y_val_t, X_t, X_t_n, Y_t, norm, args, [device_s, device])
                    
    elif args.mode == 'cr_user':
        
        if args.dataset == 'PAR':
            #user_s = ['5','6','15','3','13','14','1','7','4']
            user_s = ['5', '10', '3', '12', '13', '14', '15']
            if args.device == 'watch':
                position = 'forearm'
            elif args.device == 'phone':
                position = 'waist'
            data_s = load_data(user=user_s, position=position, sensor='acc', path=args.path)
            #data_s = data_s[data_s['activity']!='jumping']
            #user_t=['10','11', '12','9','8' ] # target users
            user_t=['1', '4', '7', '8', '9', '11' ]
        elif args.dataset == 'HAR':
            
            #pdb.set_trace()
            if args.device == 'phone':
                df = pd.read_csv(os.path.join(args.path, "Phones_accelerometer.csv"))
                position = 'samsungold'
            elif args.device == 'watch':
                df = pd.read_csv(os.path.join(args.path, "Watch_accelerometer.csv"))
                position = 'lgwatch'
                #position = 'gear'
            
            df = df.dropna()
            df.rename(columns={'x': 'attr_x', 'y': 'attr_y','z': 'attr_z','gt':'activity'}, inplace=True)
            user_t = ['e','d','f']
            
            
        for user in user_t:
            args.user = user
            if args.dataset == 'PAR':
                
                #pdb.set_trace()
                data_t = load_data(user=[args.user], position=position, sensor='acc', path = args.path)
            #data_t=data_t[data_t['activity']!='jumping']
            else:
                #pdb.set_trace()
                #all_usr = ['f','a','b','d','c','g','e','h']
                #all_usr.remove(user)
                #data_s = extract_users(df,all_usr)
                data_s = extract_users(df,['a','b','g','c','h'])
                data_s = data_s[data_s['Model'] == position]
                data_t = extract_users(df, args.user)
                data_t = data_t[data_t['Model'] == position]
            X_train_s, X_train_s_n, y_train_s, X_val_s, \
                        y_val_s, X_val_t, y_val_t, X_t, \
                          X_t_n, Y_t, norm = prep_data(data_t, data_s)
            train(X_train_s, X_train_s_n, y_train_s, X_val_s, y_val_s, 
                                X_val_t, y_val_t, X_t, X_t_n, Y_t, norm, args, [args.device, args.user])
            
    elif args.mode == 'cr_device':
        df = pd.read_csv(os.path.join(args.path, "Phones_accelerometer.csv"))       
        df = df.dropna()
        df.rename(columns = {'x': 'attr_x', 'y': 'attr_y','z': 'attr_z','gt':'activity'}, inplace=True)
        #device_ss = ['nexus4','s3','samsungold','s3mini']
        #data = extract_users(df,['f','a','b','d','c','g','e'])
        data = extract_users(df,['a','b','g','c','h'])
        devices = {'s3mini':100,'s3':150,'nexus4':200,'samsungold':50}
        
        for device_s in devices:
            for device_t in devices:
                if device_t != device_s:
                    data_s = data[data['Model'] == device_s]
                    data_t = data[data['Model'] == device_t]
                     #downsampling to 50 Hz
                    if devices[device_t] > 50:
                        data_t = down_sampling2(data_t,devices[device_t], 50)
                    if devices[device_s] > 50:
                        data_s = down_sampling2(data_s,devices[device_s], 50)
                    # if device_t!='samsungold' and device_s !='samsungold' :
                    #      data_t = down_sampling2(data_t,device_tt[device_t], 50)
                    #      data_s = down_sampling2(data_s,device_tt[device_s], 50)
                        
                       
                    # elif device_t!='samsungold' and device_s =='samsungold':
                         
                    #     data_t = down_sampling2(data_t, device_tt[device_t], 50)
                         
                    # elif device_t == 'samsungold' and device_s != 'samsungold':
                    #     data_s = down_sampling2(data_s,device_tt[device_s], 50)
                     
                    X_train_s, X_train_s_n, y_train_s, X_val_s, \
                        y_val_s, X_val_t, y_val_t, X_t, \
                          X_t_n, Y_t, norm = prep_data(data_t, data_s)
                    train(X_train_s, X_train_s_n, y_train_s, X_val_s, y_val_s, 
                                X_val_t, y_val_t, X_t, X_t_n, Y_t, norm, args, [device_s, device_t])
       
if __name__ == "__main__":
    main()
