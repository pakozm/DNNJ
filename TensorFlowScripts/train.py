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

import common
import gzip
import math
import numpy as np
import os
import tensorflow as tf
import urllib

from six.moves import xrange  # pylint: disable=redefined-builtin

MNIST_BASIC_URL = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip"
MNIST_ZIP_FILENAME = "mnist-basic.zip"

TRAIN = "mnist_train.amat"
TEST = "mnist_test.amat"

common.downloadData(MNIST_BASIC_URL, MNIST_ZIP_FILENAME)
os.system("unzip " + MNIST_ZIP_FILENAME)
  
bunch_size = 128
hidden_size = 2048
num_layers = 3
replacement = 256
rpdecay = 0.999
rpenalty = 0.2
salt = 0.2
weight_decay = 0.01
learning_rate = 1e-03
assert replacement % bunch_size == 0

np.random.seed(1234)

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


# In[6]:

mnist = read_data_sets(DATA_DIR, one_hot=True)

x  = tf.placeholder("float", shape=[None, 784])
y  = tf.placeholder("float", shape=[None, 10])
rp = tf.placeholder("float", shape=[])

# In[12]:

def weight_variable(shape, *args, **kwargs):
  initial = tf.truncated_normal(shape, stddev=0.2)
  return tf.Variable(initial, *args, **kwargs)

def bias_variable(shape, value, *args, **kwargs):
  initial = tf.constant(value, shape=shape)
  return tf.Variable(initial, *args, **kwargs)

def cross_entropy(hat_y,y):
    return tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(hat_y,y) )

def cross_entropy2(hat_y,y):
    return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(hat_y,y) )

def sdae(x, w, b1, b2, m):
    h = tf.nn.sigmoid( tf.matmul( tf.mul(x, m), w ) + b1 )
    hat_x = tf.matmul( h, tf.transpose(w) ) + b2
    return hat_x

# In[13]:

sizes = [784] + [hidden_size]*num_layers + [10]
Ws    = [ weight_variable([sizes[i],sizes[i+1]]) for i in range(len(sizes)-1) ]
bs    = [ bias_variable([1,sizes[i+1]], 0.0) for i in range(len(sizes)-1) ]
bs2   = [ bias_variable([1,sizes[i]], 0.0) for i in range(len(sizes)-1) ]

masks = [ tf.placeholder("float", shape=[None, sz]) for sz in sizes[:-1] ]

Hs = [ x ]

#with tf.name_scope("DNN") as scope:
for i in range(len(Ws)-1):
    w,b,in_x = Ws[i],bs[i],Hs[-1]
    Hs.append( tf.nn.sigmoid( tf.matmul( in_x, w ) + b ) )
hat_y = tf.matmul( Hs[-1], Ws[-1] ) + bs[-1]
Ls = cross_entropy(hat_y, y)
#tf.scalar_summary("Ls", Ls)

#with tf.name_scope("SDAEs") as scope:
Lus = [ rp*cross_entropy2(sdae(Hs[i],Ws[i],bs[i],bs2[i],masks[i]), Hs[i]) for i in range(len(Hs)) ]
#for i,Lu in enumerate(Lus): tf.scalar_summary("Lu" + str(i), Lu)

#with tf.name_scope("EmpiricalRisk") as scope:
EmpiricalRisk = Ls
for Lu in Lus: EmpiricalRisk += Lu
#with tf.name_scope("Reg.") as scope2:
#for w in Ws: EmpiricalRisk += weight_decay * tf.nn.l2_loss(w)
EmpiricalRisk += weight_decay * tf.nn.l2_loss(Ws[-1])
#    tf.scalar_summary("EmpiricalRisk", EmpiricalRisk)

#with tf.name_scope("Train") as scope:
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(EmpiricalRisk)

#with tf.name_scope("Predict") as scope:
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(hat_y),1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#tf.scalar_summary("error", 1.0 - accuracy)

#merged = tf.merge_all_summaries()

# Initializing the variables
init = tf.initialize_all_variables()


# In[ ]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #writer = tf.train.SummaryWriter("/home/experimentos/tmp/jmlr_logs", sess.graph_def)    

    best_val_acc = 0
    best_tst_acc = 0
    rp_value = rpenalty
    for i in range(4000):
        train_losses = []
        for j in range(int(math.ceil(replacement/bunch_size))):
            batch_x,batch_y = mnist.train.next_batch(bunch_size)
            feed  = { x: batch_x, y: batch_y, rp: rp_value }
            for k,sz in enumerate(sizes[:-1]):
              feed[masks[k]] = np.random.binomial(1,salt,[bunch_size,sz])
            #_,train_loss,result = sess.run([train_step,EmpiricalRisk,merged]).run(feed_dict=feed)
            _,train_loss = sess.run([train_step,EmpiricalRisk], feed_dict=feed)
            train_losses.append(train_loss)
            #summary_str = result[0]
            #writer.add_summary(summary_str, i+1)
            rp_value *= rpdecay
        train_loss = np.mean(train_losses)
        val_acc = accuracy.eval(feed_dict={x: mnist.validation.images,
                                           y: mnist.validation.labels})
        val_loss = 1.0 - val_acc
        if val_acc > best_val_acc and i > 400:
            best_val_acc = val_acc
            best_test_acc = accuracy.eval(feed_dict={x: mnist.test.images,
                                                     y: mnist.test.labels})
            print(i+1, train_loss, val_loss, 1.0 - best_test_acc)
        else:
            print(i+1, train_loss, val_loss)            
    print(1.0-best_val_acc, 1.0-best_test_acc)
