from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hickle as hkl
from functools import partial
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
import seaborn as sns
import os

#matplotlib.use("TkAgg")

def compare_recon(sess,train,model,ch,save_fig):
    s=sess.run([model.decoder_t],feed_dict = {model.X_t_n:train})
    reconstructed=np.array([i-i.mean(0) for i in s[0]])
   # reconstructed=np.array([(j-j.min(0))/(j.max(0)-j.min(0)) for j in s[0]])
    #reconstructed=s[0]
    train=np.array([n-n.mean(0) for n in train])
   # train=np.array([(j-j.min(0))/(j.max(0)-j.min(0)) for j in train])
 
    
    #plt.title('Training Source Images')
    ints=np.random.randint(0,len(train),6)
    for i in ints:
        plt.ion()
        # plot original and reconstructed
        x = np.arange(128)
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 4))
        sns.lineplot(x,reconstructed[i, :, ch], color='green', alpha=0.7, label='Reconstructed', ax=ax1)
        ax1.lines[0].set_linestyle("--")
        sns.lineplot(x,train[i, :, ch], color='red', label='Original', ax=ax1)
        
        ax1.legend(loc='upper right')
        #
        if save_fig==True:
            fig.savefig('{}.png'.format(i))
        plt.ioff()
        plt.show()




def gen_noise(shape, x, zero_data = False, scale=0.000):
    if not zero_data:
        return np.random.normal(loc=0, scale=scale, size=shape)
    # zero out some values
    zeros = np.zeros(shape)
    unif_n = np.arange(x.size)/(x.size-1)
    np.random.shuffle(unif_n)
    # percentage of frames to keep, 1-keep will be set to zero
    keep = 0.97
    mask = (unif_n>keep).reshape(shape)
    x[mask] = zeros[mask]
    # -x will zero out 1-keep % of elements
    return -x

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  
def check_num_activity(data, user):
    """
    Check the number of activities for each user in the dataframe data
    input: dataframe with loaded users
            list of users
    output: print # of activities for each user
    """
    for i in user:
        act=data[data['user']==str(i)]['activity'].value_counts()
        print('user',i,'num_activities ',act.index.shape[0])
     


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)



def normalize(x, mean = None, std = None):

    x_reshaped = tf.reshape(x,[-1,3])
  
        # mean of each column/sensor
    x_reshaped = (x_reshaped-mean)/std
    # reshape back to original shape
    return tf.reshape(x_reshaped,[-1,128,3])


def fixprob(att):
    att = att + 1e-9
    _sum = tf.reduce_sum(att, reduction_indices=1, keep_dims=True)
    att = att / _sum
    att = tf.clip_by_value(att, 1e-9, 1.0, name=None)
    return att






def save_file(file,path):
#    if not os.path.exists(path):
#        os.makedirs(path)
    hkl.dump(file,path)
    
    
def load_file(path):
    file=hkl.load(path)
    return file
