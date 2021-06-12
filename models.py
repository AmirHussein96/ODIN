from flip_gradient import flip_gradient
from utils import *
import tensorflow as tf
from tensorflow.keras import layers
from utils import batch_generator
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import os
import pdb


def fixprob(att):
    att = att + 1e-9
    _sum = tf.reduce_sum(att,reduction_indices=1, keep_dims=True)
    att = att / _sum
    att = tf.clip_by_value(att, 1e-9, 1.0, name=None)
    return att

def kl(x, y):
    x = fixprob(x)
    y = fixprob(y)
    X = tf.distributions.Categorical(probs=x)
    Y = tf.distributions.Categorical(probs=y)
    return tf.distributions.kl_divergence(X, Y)

def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
  We create a sum of multiple gaussian kernels each having a width sigma_i.
  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the MMD between two representations.
  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight
  return loss_value


def conv_layer(input,k, size_in, size_out,stride,name="conv",init_val=None,tr=True, max_pool=True):
    
  with tf.name_scope(name):
    if init_val==None:
        #w = tf.get_variable(name="W%s"%(name),shape=[k,size_in, size_out],initializer=tf.contrib.layers.variance_scaling_initializer(),trainable=tr)
        w=tf.get_variable(name="W%s"%(name),initializer=tf.truncated_normal([k,size_in, size_out], stddev=0.1),trainable=tr)
        b = tf.get_variable(initializer=tf.constant_initializer(0.1),name="B%s"%(name),shape=[size_out],trainable=tr)
        
    else:
        w = tf.get_variable(name="W%s"%(name),shape=[k,size_in, size_out],initializer=tf.constant_initializer(init_val[0]),trainable=tr)
        b = tf.get_variable(initializer=tf.constant_initializer(init_val[1]),name="B%s"%(name),shape=[size_out],trainable=tr)
    conv = tf.nn.conv1d(input, w, stride=stride,padding='SAME')
#    if batch_norm==True:
    #with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
    act = tf.nn.relu(conv + b)
       # act = tf.layers.batch_normalization(act, training=tr)
#    tf.summary.histogram("weights", w)
#    tf.summary.histogram("biases", b)
#    tf.summary.histogram("activations", act) 
    if max_pool == True:
        return  tf.layers.max_pooling1d(act, pool_size=2, strides=2, padding='same')
    else:
        return  act



def fc_layer(input, size_in, size_out, name="fc",init_val=None,tr=True):
  
  with tf.name_scope(name):
   if init_val==None:
       # w = tf.get_variable(name="W%s"%(name),shape=[size_in, size_out],initializer=tf.contrib.layers.variance_scaling_initializer(),trainable=tr)
        w = tf.get_variable(name="W%s"%(name),initializer=tf.truncated_normal([size_in, size_out], stddev=0.1),trainable=tr)
        b = tf.get_variable(initializer=tf.constant_initializer(0.1),name="B%s"%(name),shape=[size_out],trainable=tr)
        
   else:
        w = tf.get_variable(name="W%s"%(name),shape=[size_in, size_out],initializer=tf.constant_initializer(init_val[0]),trainable=tr)
        b = tf.get_variable(initializer=tf.constant_initializer(init_val[1]),name="B%s"%(name),shape=[size_out],trainable=tr)
   act = tf.matmul(input, w) + b
#    tf.summary.histogram("weights", w)
#    tf.summary.histogram("biases", b)
#    tf.summary.histogram("activations", act)
   return act
        
        
class Source_model(object):
    """Source model."""
    
    def __init__(self, batch, args, n_lables, rec_loss=0.01):
        
        self.batch_size = batch
        self.n_lables = n_lables
        self.recon_loss = rec_loss
        self._build_model()
       
    
    def _build_model(self):
        alpha=0.5 
        self.X = tf.placeholder(tf.float32, [None, 128, 3],name='input')
        self.X_n = tf.placeholder(tf.float32, [None, 128, 3],name='noisy_input')
        self.y = tf.placeholder(tf.float32, [None, self.n_lables],name='labels')
        self.train = tf.placeholder(tf.bool, [],name='train_flag')
        self.keep_rate = tf.placeholder_with_default(1.0, shape=(),name='keep_rate') 
        # CNN model for feature extraction
       
        with tf.variable_scope('source_feature_extractor'):            
            conv1 = conv_layer(self.X_n, 5, 3, 32 , stride=1, name="conv1")
            print('conv1: ',conv1.shape)
           
#            tf.summary.histogram('feature_layer1',layer1)
            conv2 = conv_layer(conv1, 3, 32, 64,stride=1, name="conv2")
            #conv2_drop = tf.nn.dropout(conv2, self.keep_rate)
            print('conv2: ',conv2.shape)
            
            conv3 = conv_layer(conv2, 3, 64, 128,stride=1, name="conv3")
            #conv2_drop = tf.nn.dropout(conv2, self.keep_rate)
            print('conv3: ',conv3.shape)
            #pdb.set_trace()
            gru, state_h=layers.GRU(128,name='gru_1',return_sequences=True,return_state=True)(conv3)
            # The domain-invariant feature
            
            print('gru_output: ',gru.shape)
            print('gru_state: ',state_h.shape)
           
            dim = conv3.get_shape().as_list()[1]*conv2.get_shape().as_list()[2]
            self.feature = gru[:,-1,:]  #state_h # gru[:,-1,:] 
            print( 'feature_shape: ',self.feature.shape)
            
            # self.feature = tf.nn.dropout(fc1_relu, self.keep_rate)
           
            # tf.summary.histogram('feature_final_layer',self.feature)
           
            # MLP for class prediction
            #print('dim',dim)
        with tf.variable_scope('label_predictor'):
            
            logits=layers.Dense(self.n_lables, name="fc2")(self.feature)
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels= self.y)
        trans1 = tf.keras.layers.Conv2DTranspose(64, (3,1),padding='same',name='conv_trans1')
        trans2 = tf.keras.layers.Conv2DTranspose(32, (3,1),padding='same',name='conv_trans2')
        trans3 = tf.keras.layers.Conv2DTranspose(3, (5,1),padding='same',name='conv_trans3')
        
        decode_upsample1=tf.keras.layers.UpSampling1D(2,name='upsample1')
        decode_upsample2=tf.keras.layers.UpSampling1D(2,name='upsample2')
        
        with tf.variable_scope('decoder'):
            print('Decoder layers shape')
            print('input: ', self.feature.shape )
            gru_s_output = layers.GRU(128, name="gru_2",return_sequences=True)(gru)
            #gru_s_output=tf.reshape(gru_s_output,[-1,30,64])
            print('gru: ',gru_s_output.shape )
            decode_upsample1_s=decode_upsample1(gru_s_output)
            print('upsample1 : ',decode_upsample1_s.shape )
            decode_upsample1_s=tf.reshape(decode_upsample1_s,[-1,32,1,128])
            #print(decode_upsample1_s.shape)
            x_s =trans1(decode_upsample1_s)
            
            #x_s=tf.nn.leaky_relu(x_s,alpha=alpha)
            x_s = tf.nn.relu(x_s)
            #print(x.shape)
            x1_s=tf.reshape(x_s,[-1,32,64])
            x2_s=decode_upsample2(x1_s)
            print('upsample2 : ',x2_s.shape )
            x2_s=tf.reshape(x2_s,[-1,64,1,64])
            decoded_s = trans2(x2_s)
            print('decoded : ',decoded_s.shape )
            #decoded_s =tf.nn.leaky_relu(decoded_s,alpha=alpha)
            decoded_s = tf.nn.relu(decoded_s)
            decoded_s=tf.reshape(decoded_s,[-1,64,32])
            decoded_s=decode_upsample2(decoded_s)
            decoded_s=tf.reshape(decoded_s,[-1,128,1,32])
            self.decoder_s = trans3(decoded_s)
            self.decoder_s=tf.reshape(self.decoder_s,[-1,128,3])
            print('decoder_s ',self.decoder_s.shape)
            
            # Build the model graph
        
        #self.learning_rate = tf.placeholder(tf.float32, [])
        self.l2_loss = 0.004*sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        self.pred_loss = tf.reduce_mean(self.pred_loss)
        #loss_summary1=tf.summary.scalar('pred_loss',self.pred_loss)
        self.recon_loss = tf.losses.mean_pairwise_squared_error(self.X, self.decoder_s)*self.recon_loss
        self.total_loss = self.pred_loss + self.recon_loss + self.l2_loss
        # optimizer
        self.train_opt = tf.train.AdamOptimizer().minimize(self.total_loss)
        # evaluation
        correct_label_pred = tf.equal(tf.argmax( self.y, 1), tf.argmax(self.pred, 1))
        self.label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        # acc_summary1=tf.summary.scalar('accuracy_on_activity', self.label_acc)
      
    def parameters(self, save=False):
            
        #%%
        #Extracting pretrained model weights
        self.feature=[]
        for i in tf.trainable_variables(scope='source_feature_extractor'):
            self.feature.append(self.sess.run(i))
        #%%
        self.label_class=[]
        for i in tf.trainable_variables(scope='label_predictor'):
            self.label_class.append(self.sess.run(i))
            
        self.decode_w=[]
        for i in tf.trainable_variables(scope='decoder'):
            self.decode_w.append(self.sess.run(i))
        if save == True:
            save_file(self.feature,'./feat_extract_weights')
            save_file(self.label_class,'./label_class_weights')
        return self.feature, self.label_class, self.decode_w
    
    def train_and_evaluate(self, graph, data, loggling, num_steps=8000, verbose=True):
        """Helper to run the model with different training modes."""

        self.num_steps = num_steps
        
        X_t, X_t_n, Y_t, X_train_s, X_train_s_n, y_train_s, X_val_t, y_val_t, X_val_s, y_val_s = data
       
        
        history = dict(source_acc=[],target_acc=[],domain_acc=[],embed=[])
        # with tf.Session() as sess:
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer()) 
        # Batch generators
        gen_source_only_batch = batch_generator(
            [X_train_s, X_train_s_n, y_train_s], self.batch_size)
      
        # Training loop
        for i in range(num_steps):
            
            X, X_n, y = next(gen_source_only_batch)
            
            _, batch_loss,p_acc,rec_loss,l2 = self.sess.run([self.train_opt, 
                                                      self.pred_loss, self.label_acc, 
                                                      self.recon_loss, self.l2_loss],
                                  feed_dict={self.X: X, self.X_n: X_n, self.y: y, self.train: False
                                          , self.keep_rate:1})
        
            if verbose and i % 1000 == 0:
                    source_acc = self.sess.run(self.label_acc,
                            feed_dict={self.X_n: X_val_s, self.y:y_val_s,
                                        self.train: False,self.keep_rate:1.0})
                    print('batch_loss: %.3f  source_val_acc: %.3f rec_loss: %.3f l2: %.3f'%(
                            batch_loss, source_acc, rec_loss, l2))
                    loggling.info('batch_loss: %.3f  source_val_acc: %.3f rec_loss: %.3f '%(
                            batch_loss, source_acc, rec_loss))
            
        # Compute final evaluation on test data
        loggling.info("Testing..........................")
        source_acc = self.sess.run(self.label_acc,
                            feed_dict={self.X_n: X_val_s, self.y:y_val_s,
                                        self.train: False,self.keep_rate:1.0})
        num_test = 400
        combined_test = np.vstack([X_val_s[:num_test], X_val_t[:num_test]])
        test_emb = self.sess.run(self.feature, feed_dict={self.X_n: combined_test, self.train: False})
       
        # saving results
        history['source_acc'].append(source_acc)
       
        history['embed'].append(test_emb)
        print('Source test accuracy:', np.array(history['source_acc']).mean())
        
        loggling.info('Source test accuracy: %.3f' %(np.array(history['source_acc']).mean()))
        
        
        return  test_emb, self.sess, history



    
class ODIN(object):
    
    """ODIN model"""
    
    def __init__(self, da_loss, parameters, n_lables, args, batch=256, rec_loss=0.005):
        self.batch = batch
        self.da_loss = da_loss
        self.n_lables = n_lables
        self.pretrained_par = parameters
        self.rec_loss = rec_loss
        self._build_model()
       
    def nonlinear_modeling(self, var1, var2, name):
   
        with tf.name_scope(name):
            
            var1=tf.reshape(var1,[-1])
            n=var1.get_shape().as_list()[0]
            var2=tf.reshape(var2,[-1])
           
            a = tf.Variable(tf.ones(n)*0.5, name="a")
           # a = tf.Variable(tf.zeros(n), name="a")
            b = tf.Variable(tf.zeros(n), name="b")
        
            A = var1-var2-tf.keras.activations.tanh(a*var2+b)
            return tf.nn.l2_loss(A)
        
        
    def _build_model(self):
        self.X_s = tf.placeholder(tf.float32, [None, 128, 3],name='input_s')
        self.X_t = tf.placeholder(tf.float32, [None, 128, 3],name='input_t')
        # pdb.set_trace()
        self.y_s = tf.placeholder(tf.float32, [None, self.n_lables],name='labels_s')
        self.y_t = tf.placeholder(tf.float32, [None, self.n_lables],name='labels_t')
        self.X_s_n = tf.placeholder(tf.float32, [None, 128, 3],name='input_s_n')
        self.X_t_n = tf.placeholder(tf.float32, [None, 128, 3],name='input_t_n')
        
        self.l = tf.placeholder(tf.float32, [],name='mmd_weight')
        self.train = tf.placeholder(tf.bool, [],name='train_flag')
        self.domain= tf.placeholder(tf.float32, [None, 2],name='domain')
        self.keep_rate=tf.placeholder_with_default(1.0, shape=(),name='keep_rate')
        
        
    #        
    #        self.X_in_s=normalize(self.X_s,data_mean,data_std)
    #        self.X_in_t=normalize(self.X_t,data_mean,data_std)
        # CNN model for feature extraction
        with tf.variable_scope('source_feature_extractor') as source:
            conv1 = conv_layer(self.X_s_n, 5 ,3, 32,stride=1, name="source_conv1",
                                init_val=[self.pretrained_par[0][0],self.pretrained_par[0][1]])
            
        with tf.variable_scope('target_feature_extractor') as target:
            conv1_t = conv_layer( self.X_t_n, 5,3, 32,stride=1, name="target_conv1",
                                  init_val=[self.pretrained_par[0][0],self.pretrained_par[0][1]])
            
        
    #            tf.summary.histogram('feature_layer1',layer1)
        with tf.variable_scope(source):
            conv2 = conv_layer(conv1, 3, 32, 64,stride=1, name="source_conv2",
                                init_val=[self.pretrained_par[0][2],self.pretrained_par[0][3]])
            
        with tf.variable_scope(target):
            conv2_t = conv_layer(conv1_t, 3, 32, 64,stride=1, name="target_conv2",
                                  init_val=[self.pretrained_par[0][2],self.pretrained_par[0][3]])
    #
          #            tf.summary.histogram('feature_layer1',layer1)
        with tf.variable_scope(source):
            conv3 = conv_layer(conv2, 3, 64, 128,stride=1, name="source_conv3",
                                init_val=[self.pretrained_par[0][4],self.pretrained_par[0][5]])
            
        with tf.variable_scope(target):
            conv3_t = conv_layer(conv2_t, 3, 64, 128,stride=1, name="target_conv3",
                                  init_val=[self.pretrained_par[0][4],self.pretrained_par[0][5]])
    #           
    
        # tf.summary.histogram('feature_final_layer',self.feature)
          
    
        gru_enc = layers.GRU(128,return_sequences=True,return_state=True,
                              name='gru_s',kernel_initializer=tf.constant_initializer(self.pretrained_par[0][6]),
                              recurrent_initializer=tf.constant_initializer(self.pretrained_par[0][7]),
                              bias_initializer=tf.constant_initializer(self.pretrained_par[0][8]))
        with tf.variable_scope('source_feature_extractor'):
            gru_s,h_s = gru_enc(conv3)
            
        with tf.variable_scope('target_feature_extractor'):   
            gru_t,h_t = gru_enc(conv3_t)
        # The domain-invariant feature
        
        self.feature_s = gru_s[:, -1, :] #h_s #gru_s[:, -1, :] 
        self.feature_t = gru_t[:, -1, :] #h_t #gru_t[:, -1, :] 
        domain_merge = layers.concatenate([self.feature_s, self.feature_t], 
                                          name='merged_features',axis=0)
        self.feature = domain_merge
        
        # MLP for class prediction
        
    #            tf.summary.histogram('feature_layer1',layer1)
            
        with tf.variable_scope('label_predictor'):
    
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            
            target_features = lambda: self.feature_t
            source_features = lambda: self.feature_s
            classify_feats = tf.cond(self.train, source_features, target_features)
    
          #  all_labels = lambda: self.y
            target_labels = lambda:self.y_t
            source_labels = lambda:self.y_s
            self.classify_labels = tf.cond(self.train, source_labels, target_labels)
            dim=classify_feats.get_shape().as_list()[1]
    #            fc1_s=fc_layer(classify_feats, dim, 64, name="source_fc1")
    #            fc1_s_relu=tf.nn.relu(fc1_s)
    #            fc1_s_relu = tf.nn.dropout(fc1_s_relu, self.keep_rate)

            logits=fc_layer(classify_feats, dim, self.n_lables, name="source_fc2",
                            init_val=[self.pretrained_par[1][0], 
                                      self.pretrained_par[1][1]], tr=False)
            
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                      labels=self.classify_labels)
    
        # Small MLP for domain prediction with mmd loss
        if self.da_loss=='MMD':
            with tf.variable_scope('MMD'):
                x = fixprob(self.feature_s)
                y = fixprob(self.feature_t)
                self.adversarial_loss = mmd_loss(x, y, self.l)
        elif self.da_loss=='DC':
            with tf.variable_scope('domain_predictor'):
          
                # Flip the gradient when backpropagating through this operation
                with tf.name_scope('Flip_gradient'):
                    feat = flip_gradient( self.feature, 1)
                
                layer3 = layers.Dense(128,activation=tf.nn.relu,name='DA_dense1')(feat)
                layer3 = tf.nn.dropout(layer3, self.keep_rate)
                layer3 = layers.Dense(64,activation=tf.nn.relu,name='DA_dense2')(layer3)
                
                d_logits = layers.Dense(2,name='domain_pred')(layer3)
              
                
                self.domain_pred = tf.nn.softmax(d_logits)
                self.adversarial_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=d_logits,
                                                            labels=self.domain))
        gru_dec = layers.GRU(128,return_sequences=True,name='gru_s',
                              kernel_initializer=tf.constant_initializer(self.pretrained_par[2][0]),
                            recurrent_initializer=tf.constant_initializer(self.pretrained_par[2][1]),
                            bias_initializer=tf.constant_initializer(self.pretrained_par[2][2]))
        trans1 = tf.keras.layers.Conv2DTranspose(64,(3,1), kernel_initializer=tf.constant_initializer(
            self.pretrained_par[2][3]),bias_initializer=tf.constant_initializer(self.pretrained_par[2][4]),
            padding='same',name='conv_trans1')
        trans2 = tf.keras.layers.Conv2DTranspose(32, (3,1),kernel_initializer=tf.constant_initializer(
            self.pretrained_par[2][5]), bias_initializer=tf.constant_initializer(self.pretrained_par[2][6]),
            padding='same',name='conv_trans2')
        trans3 = tf.keras.layers.Conv2DTranspose(3, (5,1),kernel_initializer=tf.constant_initializer(
            self.pretrained_par[2][7]),bias_initializer=tf.constant_initializer(self.pretrained_par[2][8]),
            padding='same',name='conv_trans3')
        
        decode_upsample1 = tf.keras.layers.UpSampling1D(2,name='upsample1')
        decode_upsample2 = tf.keras.layers.UpSampling1D(2,name='upsample2')
        
        with tf.variable_scope('decoder_f') as scope:
            print('Decoder layers shape')
            print('input: ', gru_s.shape )
           
            gru_s_dec=gru_dec(gru_s)
           
            print('gru_s_dec: ', gru_s_dec.shape )
            decode_upsample1_s=decode_upsample1(gru_s_dec)
            print('upsample1 : ',decode_upsample1_s.shape )
            decode_upsample1_s=tf.reshape(decode_upsample1_s,[-1,32,1,128])
            #print(decode_upsample1_s.shape)
            x_s =trans1(decode_upsample1_s)
            
            x_s=tf.nn.relu(x_s)
            #print(x.shape)
            x1_s=tf.reshape(x_s,[-1,32,64])
            x2_s=decode_upsample2(x1_s)
            print('upsample2 : ',x2_s.shape )
            x2_s=tf.reshape(x2_s,[-1,64,1,64])
            decoded_s = trans2(x2_s)
            decoded_s =tf.nn.relu(decoded_s)
            decoded_s=tf.reshape(decoded_s,[-1,64,32])
            decoded_s=decode_upsample2(decoded_s)
            print('upsample3 : ',decoded_s.shape )
            decoded_s=tf.reshape(decoded_s,[-1,128,1,32])
            #pdb.set_trace()
            self.decoder_s = trans3(decoded_s)
        
            self.decoder_s=tf.reshape(self.decoder_s, [-1,128,3])
            print('decoder_s ',self.decoder_s.shape)
    #        #target decoder
    
        with tf.variable_scope(scope,reuse=True) :
            
            gru_t_dec=gru_dec(gru_t)
            print('gru_s_dec: ',gru_t_dec.shape )
            
            decode_upsample=decode_upsample1(gru_t_dec)
            decode_upsample=tf.reshape(decode_upsample, [-1,32,1,128])
            print(decode_upsample.shape)
            x =trans1(decode_upsample)
            x=tf.nn.relu(x)
            #print(x.shape)
            x1=tf.reshape(x,[-1,32,64])
            x2=decode_upsample2(x1)
            x2=tf.reshape(x2,[-1,64,1,64])
            decoded = trans2(x2)
            decoded = tf.nn.relu(decoded)
            decoded = tf.reshape(decoded,[-1,64,32])
            decoded = decode_upsample2(decoded)
            decoded = tf.reshape(decoded,[-1,128,1,32])
            decoded = trans3(decoded)
            self.decoder_t=tf.reshape(decoded,[-1,128,3])
        
    
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.L2_loss=tf.placeholder(tf.float32, [])
        self.pred_loss = tf.reduce_mean(self.pred_loss)*0.5
        # loss_summary1=tf.summary.scalar('pred_loss',pred_loss)
        self.adver_loss =self.adversarial_loss
        # loss_summary2=tf.summary.scalar('domain_loss',domain_loss)
        
    
        # modeling the domain divergence (minimize the difference 
        #                       between parameters of the corresponding layers)
        with tf.variable_scope('loss'):   
            with tf.name_scope('total_Loss'): 
                #reconstruction loss
                target_recon_loss = tf.losses.mean_pairwise_squared_error(self.X_t, self.decoder_t)#tf.contrib.losses.mean_pairwise_squared_error(target,target_recon,1e-6)
                source_recon_loss = tf.losses.mean_pairwise_squared_error(self.X_s, self.decoder_s)#tf.contrib.losses.mean_pairwise_squared_error(source,source_recon,1e-6)# 
                self.recon_loss = (source_recon_loss + target_recon_loss) * self.rec_loss
           
                self.L2 = self.L2_loss*sum(self.nonlinear_modeling(tf_var1,tf_var2,'lin%s'%(index)) 
                                            for index, (tf_var1,tf_var2) in 
                                            enumerate(zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                                                            scope='source_feature_extractor')[0:6],
                                                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                                                            scope='target_feature_extractor')[0:6])))
                self.total_loss = self.pred_loss + self.L2 + self.adver_loss + self.recon_loss
              # tf.summary.scalar('total_Loss',pred_loss + domain_loss)
        
            
        with tf.name_scope('train'):
           
            self.adapt_opt = tf.train.AdamOptimizer(0.0005).minimize(self.total_loss)
        
        # Evaluation
        with tf.name_scope('label_acc'):
            correct_label_pred = tf.equal(tf.argmax(self.classify_labels, 1), tf.argmax(self.pred, 1))
            self.label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
            #acc_summary1 = tf.summary.scalar('accuracy_on_activity', self.label_acc)
            # acc_summary2=tf.summary.scalar('accuracy_on_activity',label_acc)


    def train_and_evaluate(self, data, graph, norm, logging=None, user=None, num_steps=1000, verbose=True):
        """ Helper to run the model with different training modes."""
        X_t, _ , Y_t, X_train_s, X_train_s_n, y_train_s, _, _ , X_val_s, y_val_s = data
        history = dict(source_acc=[], target_acc=[], domain_acc=[], embed=[], test_labels=[], test_domain=[])
    
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        fold=0
       
        for train_index, test_index in skf.split(X_t, Y_t.argmax(1)):
            sess=tf.Session()
            sess.run(tf.global_variables_initializer()) 
            
            X_train_t, X_val_t, y_train_t, y_val_t =  X_t[train_index],\
                            X_t[test_index], Y_t[train_index],Y_t[test_index]
            
            #X_t_n = X_train_t + gen_noise(X_train_t.shape, X_train_t, False)
            X_train_t = norm.transform(X_train_t.reshape(-1,3)).reshape(-1,128,3)
            X_t_n = X_train_t + gen_noise(X_train_t.shape, X_train_t, False)
            #X_t_n = norm.transform(X_t_n.reshape(-1,3)).reshape(-1,128,3)
            X_val_t = norm.transform(X_val_t.reshape(-1,3)).reshape(-1,128,3)
            num_test = 400
            combined_test_labels = np.vstack([y_val_s[:num_test], y_val_t[:num_test]])
            combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),np.tile([0., 1.], [num_test, 1])])
           
            # writer=tf.summary.FileWriter(log_dir,sess.graph)
            # initialize model parameters with pretrained 
            # init_load(sess,label_class,'label_predictor')
            #init_load(sess,feature,'source_feature_extractor')
            # init_load(sess,feature,'target_feature_extractor')
        
            # Batch generators
            gen_source_batch = batch_generator(
                [X_train_s, X_train_s_n , y_train_s], self.batch // 2)
            gen_target_batch = batch_generator(
                [X_train_t, X_t_n, y_train_t], self.batch // 2)
        
            domain_labels = np.vstack([np.tile([1., 0.], [self.batch // 2, 1]),
                                        np.tile([0., 1.], [self.batch // 2, 1])])
        
            
            
            # Training loop
                
            print('Fold: ',fold, 'Target: ',user )
            logging.info('Fold: %s Target: %s'%(fold,user ))
            fold+=1
            step = 0
            for i in range(num_steps):
             
                X0, X0_n, y0 = next(gen_source_batch)
                X1, X1_n, y1 = next(gen_target_batch)
              
                _, batch_loss, ploss, p_acc,l2,da_loss,ae_loss = sess.run(
                    [ self.adapt_opt, self.total_loss, self.pred_loss, 
                      self.label_acc,self.L2,self.adver_loss,self.recon_loss],
                    feed_dict={self.X_s: X0, self.X_s_n: X0_n, self.X_t: X1, self.X_t_n: X1_n,self.y_s: y0, 
                                self.y_t:y1,self.domain:domain_labels,
                                self.train: True, self.l: 2, self.keep_rate:1, self.L2_loss:0.5})
                
                step+=1
                if verbose and step % 100 == 0:
                    source_acc = sess.run(self.label_acc,
                                feed_dict={self.X_s_n: X_val_s[0:y_val_t.shape[0]],
                                            self.X_t_n: X_val_t, self.y_s:y_val_s[0:y_val_t.shape[0]],
                                            self.y_t:y_val_t, self.train: True, self.keep_rate:1.0})
                    #print_a_b(sess)
                    logging.info('total_loss: %.3f  source_val_acc: %.3f ploss: %.3f  adver_loss: %.3f  l2: %.3f  recon_loss: %.3f'%(
                            batch_loss, source_acc, ploss, da_loss,l2,ae_loss))
                    print('total_loss: %.3f  source_val_acc: %.3f ploss: %.3f  adver_loss: %.3f  l2: %.3f  recon_loss: %.3f'%(
                            batch_loss, source_acc, ploss, da_loss,l2,ae_loss))
                                 
            # Compute final evaluation on test data
            source_acc = sess.run(self.label_acc,
                                feed_dict={self.X_s_n: X_val_s[0:y_val_t.shape[0]],
                                            self.X_t_n: X_val_t, self.y_s:y_val_s[0:y_val_t.shape[0]],
                                            self.y_t:y_val_t, self.train: True,self.keep_rate:1.0})
            
            target_acc = sess.run(self.label_acc,
                                feed_dict={self.X_s_n:np.zeros( X_val_t.shape),
                                            self.X_t_n: X_val_t,self.y_s:np.zeros(y_val_t.shape),
                                            self.y_t:y_val_t, self.train: False,self.keep_rate:1.0})
            
            
            test_emb = sess.run(self.feature, feed_dict={self.X_s_n: X_val_s[:num_test],
                                                          self.X_t_n:X_val_t[:num_test], self.train: False})
            
            
            #saving results
            history['source_acc'].append(source_acc)
            history['target_acc'].append(target_acc)
            history['embed'].append(test_emb)
            history['test_labels'].append(combined_test_labels)
            history['test_domain'].append(combined_test_domain)
            print('Source  accuracy:', source_acc)
            print('Target  accuracy:', target_acc)
            logging.info('Source accuracy: %.3f' %(source_acc))
            logging.info('Target accuracy: %.3f' %(target_acc))
        print('Source  avg accuracy:', np.array(history['source_acc']).mean())
        print('Target  avg accuracy:', np.array(history['target_acc']).mean())
        logging.info('Source test accuracy: %.3f' %(np.array(history['source_acc']).mean()))
        logging.info('Target accuracy: %.3f' %(np.array(history['target_acc']).mean()))
        return  test_emb, sess, history, combined_test_labels, combined_test_domain, X0
   
 


    
    
    
