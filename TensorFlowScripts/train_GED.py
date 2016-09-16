# coding: utf-8

# This file is part of https://github.com/pakozm/DNNJ tool
# Copyright (c) 2016 Francisco Zamora-Martinez
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(sys.argv[0]))

import common
import getopt
import math
import numpy as np
import tensorflow as tf
import urllib

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.framework import ops

MNIST_BASIC_URL = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip"
MNIST_ZIP_FILENAME = "mnist-basic.zip"

DATA_DIR = "./"
TRAIN = "mnist_train.amat"
TEST = "mnist_test.amat"

common.downloadData(MNIST_BASIC_URL, MNIST_ZIP_FILENAME)
if not os.path.exists(TRAIN): os.system("unzip " + MNIST_ZIP_FILENAME)

mnist = None
bunch_size = 128
hidden_size = 2048
num_hidden_layers = 3
replacement = 256
gamma = 0.999 # exponential decay of unsupervised losses coefficients
Lambda = 0.0  # initial value of unsupervised losses coefficients
salt = 0.2
weight_decay = 1.0e-9
learning_rate = 1.0
momentum = 0.9
max_grads_norm = 6.5
epsilon = 1e-8
use_full_sdae_criterion = False
seed1 = 1234
seed2 = 5678
max_epochs = 2000
assert replacement % bunch_size == 0

def configure(opts):
  print(opts)
  for opt, arg in opts.iteritems():
    if opt == '-h':
      #print 'test.py -i <inputfile> -o <outputfile>'
      sys.exit()
    elif opt in ("-l", "--lr", "lr"):
      global learning_rate
      learning_rate = float(arg)
    elif opt in ("-w", "--wd", "wd"):
      global weight_decay
      weight_decay = float(arg)
    elif opt in ("-s", "--hsize", "hsize"):
      global hidden_size
      hidden_size = int(arg)
    elif opt in ("-n", "--nlayers", "nlayers"):
      global num_hidden_layers
      num_hidden_layers = int(arg)
    elif opt in ("-L", "--Lambda", "Lambda"):
      global Lambda
      Lambda = float(arg)
    elif opt in ("-g", "--gamma", "gamma"):
      global gamma
      gamma = float(arg)
    else:
      raise ValueError("Unknown option: " + opt)


def parseArgs(argv):
  try:
    opts, args = getopt.getopt(argv,"hl:w:s:n:L:g:",["lr=","wd=","hsize=","nlayers=","Lambda=","gamma="])
  except getopt.GetoptError:
    #print 'test.py -i <inputfile> -o <outputfile>'
    sys.exit(2)
  return { opt:arg for opt,arg in opts }

def minimize_and_clip_by_norm(optimizer, objective, var_list, clip_value):
  """Uses global norm for gradient clipping, which is the proper way (instead of
  using tf.clip_by_norm)"""
  gradients = optimizer.compute_gradients(objective, var_list=var_list)
  clipped_list,global_norm = tf.clip_by_global_norm(zip(*gradients)[0], clip_value)
  return optimizer.apply_gradients(zip(clipped_list, var_list))

# In[2]:

def scalar_to_one_hot(labels_scalar, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_scalar.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_scalar.ravel()] = 1
  return labels_one_hot


# In[3]:

def extract_images_and_labels(filename, one_hot=False):
  print('Extracting', filename)
  data_and_labels = np.loadtxt(filename, dtype=np.float32)
  data = data_and_labels[:,:-1]
  labels = data_and_labels[:,-1].reshape(data.shape[0],1).astype(np.int32)
  if one_hot: labels = scalar_to_one_hot(labels)
  #num_images = data.shape[0]
  #sz = math.sqrt(data.shape[1])
  #data = data.reshape(num_images, sz, sz)
  return data,labels


# In[4]:

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert len(images.shape) != 4 or images.shape[3] == 1
      if len(images.shape) != 2:
          images = images.reshape(images.shape[0],
                                  images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      # images = images.astype(np.float32)
      # images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    batch_idx = np.random.randint(self._num_examples,size=batch_size)
    batch_x = self._images[batch_idx,:]
    batch_y = self._labels[batch_idx,:]
    return batch_x,batch_y


# In[5]:

def read_data_sets(train_dir, fake_data=False, one_hot=False):

  class DataSets(object):
    pass

  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
    data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
    return data_sets

  VALIDATION_SIZE = 2000
  train_images,train_labels = extract_images_and_labels(train_dir + "/" + TRAIN,
                                                        one_hot)
  test_images,test_labels = extract_images_and_labels(train_dir + "/" + TEST,
                                                      one_hot)
  TRAIN_SIZE = train_images.shape[0] - VALIDATION_SIZE
  validation_images = train_images[TRAIN_SIZE:]
  validation_labels = train_labels[TRAIN_SIZE:]
  train_images = train_images[:TRAIN_SIZE]
  train_labels = train_labels[:TRAIN_SIZE]
  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)
  return data_sets

def weight_variable(name, shape, *args, **kwargs):
  # initial = tf.truncated_normal(shape, stddev=0.2)
  # return tf.Variable(initial, *args, **kwargs)
  # fan_sum = shape[0] + shape[1]
  # low = -4*np.sqrt(6.0/fan_sum) # use 4 for sigmoid, 1 for tanh activation 
  # high = 4*np.sqrt(6.0/fan_sum)
  # return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
  return tf.get_variable(name, shape=shape,
                         initializer=tf.contrib.layers.xavier_initializer(*args, **kwargs))


def bias_variable(name, shape, value, *args, **kwargs):
  # initial = tf.constant(value, shape=shape)
  # return tf.Variable(initial, *args, **kwargs)
  return tf.get_variable(name, initializer = tf.constant(value, shape=shape))

def softmax_cross_entropy(hat_y,y):
  return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(hat_y,y) )

def sigmoid_cross_entropy(hat_y,y):
  return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(hat_y,y) )

def dae(x, w, b1, b2, m):
  h = tf.nn.sigmoid(tf.matmul( tf.mul(x, m), w) + b1)
  hat_x = tf.matmul(h, tf.transpose(w)) + b2
  return hat_x

def main(opts):
  # In[6]:

  configure(opts)

  ops.reset_default_graph()
  np.random.seed(seed1)
  tf.set_random_seed(seed2)
  
  # global_step = tf.Variable(0, name='global_step', trainable=False)
  x           = tf.placeholder("float", name="x", shape=[None, 784])
  y           = tf.placeholder("float", name="y", shape=[None, 10])
  lambda_k    = tf.placeholder("float", name="lambda_k", shape=[])
  tf.scalar_summary("lambda_k", lambda_k)
  # tf.scalar_summary("step", global_step)

  # In[12]:


  # In[13]:

  sizes = [784] + [hidden_size]*num_hidden_layers + [10]
  Ws    = [ weight_variable("w" + str(i+1), [sizes[i],sizes[i+1]]) for i in xrange(num_hidden_layers+1) ]
  bs    = [ bias_variable("b" + str(i+1), [1,sizes[i+1]], 0.0) for i in xrange(num_hidden_layers+1) ]
  if Lambda>0.0:
    bs2   = [ bias_variable("b2_" + str(i), [1,sizes[i]], 0.0) for i in xrange(num_hidden_layers) ]

  assert len(Ws) == (len(sizes)-1)

  if Lambda>0.0:
    masks = [ tf.placeholder("float", name="mask_"+str(i), shape=[None, sz]) for i,sz in enumerate(sizes[:-2]) ]
  else:
    masks = []

  Hs = [ x ]

  with tf.name_scope("DNN") as scope:
    for i in xrange(len(Ws)-1):
      w,b,in_x = Ws[i],bs[i],Hs[-1]
      Hs.append( tf.nn.sigmoid( tf.matmul( in_x, w ) + b ) )
    hat_y = tf.matmul( Hs[-1], Ws[-1] ) + bs[-1]

  with tf.name_scope("Supervised") as scope:
    Ls = softmax_cross_entropy(hat_y, y)
    tf.scalar_summary("L_s", Ls)

  if Lambda>0.0:
    with tf.name_scope("DAEs") as scope:
      DAEs = [ dae(Hs[i],Ws[i],bs[i],bs2[i],masks[i]) for i in xrange(len(Hs)-1) ]

  if Lambda>0.0 and use_full_sdae_criterion:
    with tf.name_scope("SDAE") as scope:
      prev = x
      masks.append( tf.placeholder("float", name="mask_"+str(len(masks)),
                                   shape=[None, prev.get_shape().as_list()[1]]) )
      prev = tf.mul(prev, masks[-1])
      for w,b in zip(Ws[:-1],bs[:-1]):
        prev = tf.nn.sigmoid( tf.matmul( prev, w ) + b )
      for w,b in zip(reversed(Ws[1:-1]), reversed(bs2[1:])):
        prev = tf.nn.sigmoid( tf.matmul( prev, tf.transpose(w) ) + b )
      SDAE_output = tf.matmul( prev, tf.transpose(Ws[0]) ) + bs2[0]

  if Lambda>0.0:
    with tf.name_scope("Unsupervised") as scope:
      Lus = [ sigmoid_cross_entropy(DAEs[i], Hs[i]) for i in xrange(len(DAEs)) ]
      for i,Lu in enumerate(Lus): tf.scalar_summary("L_u" + str(i+1), Lu)
      lambda_k_Lus = [ lambda_k*Lu for Lu in Lus ]
      for i,lambda_Lu in enumerate(lambda_k_Lus): tf.scalar_summary("lambda_k_Lu" + str(i+1), lambda_Lu)
      if use_full_sdae_criterion:
        SDAE_Lu = sigmoid_cross_entropy(SDAE_output, x)
        tf.scalar_summary("L_u_SDAE", SDAE_Lu)
        lambda_k_SDAE_Lu = lambda_k * SDAE_Lu
        tf.scalar_summary("lambda_k_SDAE_Lu", lambda_k_SDAE_Lu)
        Lus.append( lambda_k_SDAE_Lu )

  with tf.name_scope("EmpiricalRisk") as scope:
    EmpiricalRisk = Ls + weight_decay*tf.nn.l2_loss(Ws[-1])
    if Lambda>0.0: EmpiricalRisk += tf.add_n(lambda_k_Lus)
    tf.scalar_summary("EmpiricalRisk", EmpiricalRisk)

  with tf.name_scope("Train") as scope:
    train_step = minimize_and_clip_by_norm(tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                                      momentum=momentum,
                                                                      use_nesterov=True),
                                           EmpiricalRisk,
                                           tf.trainable_variables(),
                                           max_grads_norm)
                                           #global_step)
                                           #.minimize(EmpiricalRisk)

  with tf.name_scope("Predict") as scope:
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(hat_y),1), tf.argmax(y,1))
    cls_error = 1.0 - tf.reduce_mean(tf.cast(correct_prediction, "float"))
    cls_error_summary = tf.scalar_summary("classification_error", cls_error)

  # Initializing the variables and the summaries
  init = tf.initialize_all_variables()
  merged_summaries = tf.merge_all_summaries()

  # In[ ]:
  
  global mnist
  if mnist is None: mnist = read_data_sets(DATA_DIR, one_hot=True)

  # In[ ]:

  # Launch the graph
  with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.train.SummaryWriter("/tmp/jmlr_logs/train", sess.graph)
    val_writer   = tf.train.SummaryWriter("/tmp/jmlr_logs/val", sess.graph)
    #test_writer  = tf.train.SummaryWriter("/tmp/jmlr_logs/test", sess.graph)

    best_val_loss  = float("+inf")
    best_tst_loss  = float("+inf")
    lambda_k_value = Lambda
    current_step   = 0
    for i in xrange(max_epochs):
      train_losses = []
      for j in xrange(int(math.ceil(replacement/bunch_size))):
        batch_x,batch_y = mnist.train.next_batch(bunch_size)
        feed = { x: batch_x, y: batch_y, lambda_k: lambda_k_value }
        for m in masks: feed[m] = np.random.binomial(1,salt,[bunch_size,m.get_shape().as_list()[1]])
        #_,train_loss,result = sess.run([train_step,EmpiricalRisk,merged_summaries]).run(feed_dict=feed)
        _,train_loss,summary = sess.run([train_step,EmpiricalRisk,merged_summaries], feed_dict=feed)
        train_losses.append(train_loss)
        current_step += 1
        train_writer.add_summary(summary, current_step)
        lambda_k_value *= gamma
      train_loss = np.mean(train_losses)
      feed = {x: mnist.validation.images, y: mnist.validation.labels}
      val_loss,summary = sess.run([cls_error,cls_error_summary], feed_dict=feed)
      val_writer.add_summary(summary, current_step)
      if val_loss < best_val_loss and i > 400:
        best_val_loss = val_loss
        # feed = {x: mnist.test.images, y: mnist.test.labels}
        # best_test_loss,summary = sess.run([cls_error,cls_error_summary], feed_dict=feed)
        # test_writer.add_summary(summary, current_step)
        print(i+1, train_loss, val_loss)
      else:
        print(i+1, train_loss, val_loss)

    return train_loss,best_val_loss

if __name__ == "__main__":
  main(parseArgs(sys.argv[1:]))
