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
from sklearn.model_selection import KFold, StratifiedKFold
from utils import *
from models import mmd_loss, fixprob, Source_model, ODIN
from data_preparation import *
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import pdb

#tf.disable_v2_behavior()
sns.set(style='whitegrid', palette='deep', font_scale=1.5)

#import random
#random.seed(5000)
#np.random.seed(5000)
#tf.set_random_seed(5000)

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#%%
# Logginf directory to save the information from the network for tensorboard
log_dir = './log'
save_path = "results/inter_comparison/MMD_AE"
if not os.path.isdir(save_path):
    os.makedirs(save_path, exist_ok=True)

# cross user cross device

#%% Loading and Normalizing the data  

def prep_data(data_t, data_s):
    """
    Normalizaing and splitting the data for training and adaptation
    """        
    X_t,Y_t=windowing(data_t)
    X_s,Y_s=windowing(data_s)  

    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split( X_s, Y_s, test_size=0.20,random_state=42,stratify=Y_s)
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_t, Y_t, test_size=0.20,random_state=42,stratify=Y_t) 
    #
    norm = StandardScaler()
    norm.fit(np.vstack([X_train_s,X_train_t]).reshape(-1,3))
    X_train_s=norm.transform(X_train_s.reshape(-1,3)).reshape(-1,128,3)
    X_train_t=norm.transform(X_train_t.reshape(-1,3)).reshape(-1,128,3)
    X_val_s=norm.transform(X_val_s.reshape(-1,3)).reshape(-1,128,3)
    X_val_t=norm.transform(X_val_t.reshape(-1,3)).reshape(-1,128,3)
    
    num_test = 400
    
    combined_test = np.vstack([X_val_s[:num_test], X_val_t[:num_test]])

    return X_train_s, y_train_s, combined_test, X_val_s, y_val_s, X_val_t, y_val_t, X_t, Y_t,norm

def get_args():
    # Get some basic command line arguements
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', 
                        help='Training mode (cr_user, cr_device, cr_user_device)', 
                        type=str, default='cr_user_device')
    parser.add_argument('-u', '--user', 
                        help='Target participant', 
                        type=str, default=None)
    parser.add_argument('-s', '--steps', help='Training steps', 
                        type=int, default=6000)
    parser.add_argument('-b', '--batch', help='Training batch size', 
                        type=int, default=128)
    parser.add_argument('-d', '--dataset', help='Dataset', 
                        type=str, default='HAR')
    parser.add_argument('-c', '--device', help='Wearable device [watch|phone]', 
                        type=str, default='watch')
    parser.add_argument('-p', '--path', help='Dataset path', 
                        type = str, default='path/to/dataset')
    
    return parser.parse_args()


def train(X_train_s, y_train_s, combined_test, X_val_s, y_val_s,
          X_val_t, y_val_t, X_t,Y_t, norm, args, device=None):
    n_lables = y_val_t.shape[1]
    batch_size = args.batch
        
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    with graph.as_default():
        data = [X_train_s, y_train_s, combined_test, 
                X_val_s, y_val_s, X_val_t, y_val_t] 
        source_model = Source_model(batch_size, args, n_lables)
        print('\n Training only source model \n')
        
        source_only_emb,sess,hist = \
            source_model.train_and_evaluate(graph, data, 
                                            num_steps=args.steps, 
                                            verbose=True)
        feature, label_class, decode_w = source_model.parameters()
        pretrained_par = [feature, label_class, decode_w]
        source_model.sess.close()
        
    tf.reset_default_graph()
    graph = tf.get_default_graph()
    with graph.as_default():  
        data = [X_t, Y_t, X_train_s, y_train_s, X_val_s, y_val_s]
        odin = ODIN('MMD', pretrained_par, n_lables, args, 256)
        if args.user != None:
             print('\n Adaptation to user %s'+ args.user)
        else: 
             print('\n Adaptation to device %s'+ device)
        test_emb, sess, history, combined_test_labels, combined_test_domain, X0 = \
                odin.train_and_evaluate(data, graph, norm, 
                                        args.user)
        save_path = os.path.join("results/", args.mode)
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        if args.user != None:
              path = os.path.join(str(save_path),'%s.hkl'%(args.user))
        else: 
              path = os.path.join(str(save_path),'%s.hkl'%(device))
        save_file(history, path)
        
        
def main():
    
    args = get_args()
    dataset = args.dataset
    if args.mode == 'cr_user_device':
        # Loading the data
        df = pd.read_csv(os.path.join(args.path, "Phones_accelerometer.csv"))
        df = df.dropna()
        df.rename(columns={'x': 'attr_x', 'y': 'attr_y','z': 'attr_z','gt':'activity'}, inplace=True)
        device_ss = ['nexus4','s3','samsungold','s3mini']
        #device_ss=['samsungold']
        device_tt = {'s3mini':100,'s3':150,'nexus4':200,'samsungold':50}
        #device_tt={'s3mini':100}
        user_t = args.user                        # change the user target
        for device_s in device_ss: 
            data_s = extract_users(df,['a','b','c','d','i'])
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
                        
                    X_train_s, y_train_s, combined_test, X_val_s, \
                        y_val_s, X_val_t, y_val_t, X_t, \
                            Y_t, norm = prep_data(data_t, data_s)
                            
                   # training and adapting
                   
                    train(X_train_s, y_train_s, combined_test, X_val_s, y_val_s, 
                                X_val_t, y_val_t, X_t,Y_t, norm, args)
                    
    elif args.mode == 'cr_user':
        
        if args.dataset == 'PAR':
            user_s = ['5','10','3','12','13','14','15']
            if args.device == 'watch':
                position = 'forearm'
            elif args.device == 'phone':
                position = 'waist'
            data_s = load_data(user=user_s, position=position, sensor='acc', path=args.path)
            # data_s = data_s[data_s['activity']!='jumping']
            user_t=[
                    '1',
                    '11',
                    '9',
                    '6',
                    '8',
                    '7'] # target users
        elif args.dataset == 'HAR':
            if args.device == 'phone':
                df = pd.read_csv(os.path.join(args.path, "Phones_accelerometer.csv"))
                device_s = 'samsungold'
            elif args.device == 'watch':
                df = pd.read_csv(os.path.join(args.path, "Watch_accelerometer.csv"))
                device_s = 'gear'
            
            df = df.dropna()
            df.rename(columns={'x': 'attr_x', 'y': 'attr_y','z': 'attr_z','gt':'activity'}, inplace=True)
            data_s = extract_users(df,['a','b','c','d','i'])
            user_t = ['e','g','h']
            data_t = extract_users(df, user_t)
            device_s='samsungold'
            data_s=data_s[data_s['Model']==device_s]
            data_t=data_t[data_t['Model']==device_s]
            
        for user in user_t:
            args.user = user
            data_t = load_data(user=user_t, position='forearm', sensor='acc', path= args.path)
           # data_t=data_t[data_t['activity']!='jumping']
            X_train_s, y_train_s, combined_test, X_val_s, \
                        y_val_s, X_val_t, y_val_t, X_t, \
                            Y_t, norm = prep_data(data_t, data_s)
            train(X_train_s, y_train_s, combined_test, X_val_s, y_val_s, 
                                X_val_t, y_val_t, X_t,Y_t, norm, args)
            
    elif args.mode == 'cr_device':
        df = pd.read_csv(os.path.join(args.path, "Phones_accelerometer.csv"))       
        df = df.dropna()
        df.rename(columns={'x': 'attr_x', 'y': 'attr_y','z': 'attr_z','gt':'activity'}, inplace=True)
        device_ss = ['nexus4','s3','samsungold','s3mini']
        data = extract_users(df,['a'])
        device_tt = {'s3mini':100,'s3':150,'nexus4':200,'samsungold':50}
        
        for device_s in device_ss:
            for device_t in device_tt:
                if device_t != device_s:
                    data_s = data[data['Model']==device_s]
                    data_t = data[data['Model']==device_t]
                     #downsampling to 50 Hz
                    if device_t!='samsungold' and device_s !='samsungold' :
                         data_t=down_sampling2(data_t,device_tt[device_t], 50)
                         data_s=down_sampling2(data_s,device_tt[device_s], 50)
                        
                       
                    elif device_t!='samsungold' and device_s =='samsungold':
                         
                        data_t=down_sampling2(data_t,device_tt[device_t], 50)
                         
                    elif device_t=='samsungold' and device_s !='samsungold':
                        data_s=down_sampling2(data_s,device_tt[device_s], 50)
                     
                    X_train_s, y_train_s, combined_test, X_val_s, \
                        y_val_s, X_val_t, y_val_t, X_t, \
                            Y_t, norm = prep_data(data_t, data_s)
                    train(X_train_s, y_train_s, combined_test, X_val_s, y_val_s, 
                                X_val_t, y_val_t, X_t,Y_t, norm, args, device_t)
       
if __name__ == "__main__":
    main()
