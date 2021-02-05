from scipy import stats
import pandas as pd
from os import listdir
import os
import numpy as np
from scipy.interpolate import interp1d,RegularGridInterpolator

#loading the data
def load_data(user, position, sensor, path):
        
    data1 = list()
    for i in user:
        directory = os.path.join(path,'proband%s/data/'%(i))
        activities = list()
        for name in os.listdir(directory):
            if not name.endswith(position+'.csv') :
                continue
            if not name.startswith(sensor):
                continue
            df = pd.read_csv(directory+name)
            df = df.drop(['id'], axis=1)
            columns=df.columns
            #check for null values
            if df.isnull().values.any():
                print('Null values:',df.isnull().sum())
            act=name.split('_')[1]#getting the activity
            values = df
            values['activity']=act
            values['user']=i
            activities.append(values)
        d=pd.concat(activities)
    
        data1.append(d)
    data=pd.concat(data1,)    
    return data

def extract_users(data, user):
    """This is a helping function to extract users from dataframe for HARR dataset """
    users=[]
    for i in user:
        users.append(data[data['User']==i])
    x=pd.concat(users)
    return x

#%% check number of activities for each user    
def check_num_activity(data,user):
    """
    Check the number of activities for each user in the dataframe data
    input: dataframe with loaded users
            list of users
    output: print # of activities for each user
    """
    for i in user:
        act=data[data['user']==str(i)]['activity'].value_counts()
        print('user',i,'num_activities ',act.index.shape[0])
     

#%%
def windowing(data,N_TIME_STEPS = 128,N_FEATURES = 3,step = int(128/2)):
    """
    Segmenting the time series
    input: 
            dataframe of time series
            segment size
            number of features for each segment
    outputs:
            segmented time series of shape=[segments,segment_len,features]
            one hot encoded lables
    """
    segments = []
    labels1 = []
    for i in range(0, len(data) - N_TIME_STEPS, step):       #range from 0 to length of df-batch_size that increases each time by step
        xs = data['attr_x'].values[i: i + N_TIME_STEPS]
        ys = data['attr_y'].values[i: i + N_TIME_STEPS]
        zs = data['attr_z'].values[i: i + N_TIME_STEPS]
        label = stats.mode(data['activity'][i: i + N_TIME_STEPS])[0][0] #Returns an array of the modal (most common) value in the passed array
        segments.append([xs, ys, zs])
        labels1.append(label)
    reshaped_segments = np.array(segments, dtype = np.float32).transpose(0,2,1)
    labels = np.asarray(pd.get_dummies(labels1), dtype = np.float32)
    return reshaped_segments,labels

def down_sampling(sig,input_fs,output_fs):
    n=sig.shape[0]
    sig=sig.reshape(-1,3)
    Fs=input_fs
    N=sig.shape[0]
    T=N/Fs
    scale = output_fs / input_fs
    re_n= int(scale*N)
    x = np.linspace(0.0, T, N)
   
    sig_new=np.zeros([re_n,3])
    for i in range(sig.shape[1]):
    
    
        f1 = interp1d(x, sig[:,i])
        xnew = np.linspace(0, T, num=re_n, endpoint=True) 
    
    

        sig_new[0:re_n,i]=f1(xnew)
        
       
    return sig_new.reshape(n,-1,3)

def down_sampling2(df, input_fs, output_fs):
    """

    Parameters
    ----------
    df : dataframe
       input dataframe of time series
    input_fs : int
        input frequency
    output_fs : int
        output frequency

    Returns
    -------
    dataframe with downsampled output_fs freq 

    """
    
    if input_fs==output_fs:
         return df
    else:
        sig=df[['attr_x','attr_y','attr_z']].values
       
        y=df['activity'].values
        Fs=input_fs
        N=sig.shape[0]
        T=N/Fs
        scale = output_fs / input_fs
        s=int(1/scale)
       
        re_n= int(scale*N)
        x = np.linspace(0.0, T, N)
       
        sig_new=np.zeros([re_n,3])
        for i in range(sig.shape[1]):
            f1 = interp1d(x, sig[:,i])
            xnew = np.linspace(0, T, num=re_n, endpoint=True) 
        
        
    
            sig_new[0:re_n,i]=f1(xnew)
        y_new=np.array([y[i*s:(i+1)*s][0] for i in range(y.shape[0]) if i <re_n]) 
        
        y_new=y_new.reshape(-1,1)
        
        
        df_new = pd.DataFrame(sig_new,columns = ['attr_x','attr_y','attr_z'])
        df_new['activity']=y_new
    
    return df_new