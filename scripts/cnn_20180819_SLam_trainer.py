# cnn_20180819.py

# Date:         20180819
# Description:  Tensorflow CNN Trainer (can import and save model)(S and lambda predictions)
# Engineer:     Tom Looby

from __future__ import absolute_import, division, print_function
import os
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import array
from numpy import genfromtxt
from numpy import unravel_index
import functools
import time
import datetime as dt
#import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()
import csv


start_time = time.time()
print("\nTensorFlow version: {}\n".format(tf.VERSION))

#=======================================================================
#                       Constants, User Inputs
#=======================================================================
# Number of thermocouples
tcN = 5.0
#Number of Machine Specs
machN = 4
# Number of timesteps
timeN = 48.0
# Number of elements in TC X TIME matrix
N = tcN * timeN
# Number of bins (classes) for answer
outN = 2
# Number of training datasets - goes from (experfirst, experN) 
experN = 7200
# Number to begin training on
experfirst = 1
# Number of test datasets - goes from (experN, experN + testN)
testN = 844
# Number of Epochs
num_epochs = 500000
# Batch Size
batch_size = 20
# Learning Rate (1e-3 is good)
lr = 1e-4
# Feature Map Number
fmapN = 16
# Number of Neurons per fully connected layer
neuronN = 32
# How often to record for plotting
samplelen = 5000
# How often to print to screen
printlen = 5000

# Acceptable Error threshold for S [mm]
error_thresh_S = 0.000075/2.0 #  (0.5% of range)
#error_thresh_S = 0.000075 #  (1% of range)
#error_thresh_S = 0.000225 #  (3% of range)
# error_thresh_S = 0.000377 # (5% of range)

# Acceptable Error threshold for Lambda [mm]
error_thresh_lam = 0.000244/2.0 #  (0.5% of range)
#error_thresh_lam = 0.000244 #  (1% of range)
#error_thresh_lam = 0.000732 #  (3% of range)
# error_thresh_lam = 0.00122 # (5% of range)

# Acceptable Accuracy threshold
acc_thresh = 99.0
# Moving Average Boxcar Window Length
box_window_size = 100.0


#Counter
counter = 0
#Root directory where we are working
#root_dir = '/home/workhorse/school/grad/masters/tensorflow/data_20s_nosweep_allrandom/'
root_dir = '/home/workhorse/school/grad/masters/tensorflow/data_nosweep_all/'

#Path for saving plots
figure_dir = '/home/workhorse/school/grad/masters/tensorflow/figures/20s_nosweep_allrandom'
#figure_dir = '/home/workhorse/school/grad/masters/tensorflow/figures/20s_Cs_const'

#Path for saving weights
weights_dir = '/home/workhorse/school/grad/masters/tensorflow/weights/'


#===Importing Models and Weights===
#Set this flag to 1 for importing model, 0 to start from scratch
import_flag = 1

#Original Test Model
#import_model = '20180625--21_10_24'
import_model = '20180824--01_54_53'
weight_path = '/home/workhorse/school/grad/masters/tensorflow/weights/' + import_model + '/weights'

#Path for Graph Meta Data (only sometimes used here)
meta_path = '/home/workhorse/school/grad/masters/tensorflow/weights/' + import_model + '/weights.meta'



truth = []
predicted = []
train_loss_epoch = []
train_loss_results = []
train_error = np.zeros((num_epochs//samplelen, outN))
test_error = np.zeros((num_epochs//samplelen, outN))
train_errorc1 = []
train_errorc2 = []
train_errorc3 = []
train_errorc4 = []
test_loss_results = []
train_acc = []
test_acc = []



Bp = []
P = []
freq = []
fx = []

C_range = np.zeros((4, 4), dtype=np.float32)
C_range[0][0] = 1.0/0.2
C_range[1][1] = 1.0/1.5
C_range[2][2] = 1.0/0.35
C_range[3][3] = 1.0/0.7

#Lambda Range (technically in mm but whatever)
slam_range = np.zeros((2, 2), dtype=np.float32)
slam_range[0][0] = 1.0/7.53954
slam_range[1][1] = 1.0/24.4

#err_idx = []
#=======================================================================
#                       Import Dataset
#=======================================================================
def import_data(start_ind, stop_ind):
   """
   This function reads CSV files within the input parameter bounds and
   returns a TF dataset object that includes TC data and machine specs.
   """
   TC_parms = np.zeros((int(timeN), 5))
   TC_data = np.zeros(((stop_ind - start_ind), int(timeN), int(tcN), 1))
   mach_data = np.zeros(((stop_ind - start_ind), machN))
   eich_data = np.zeros(((stop_ind - start_ind), outN))
   
   #Read data into numpy array
   for i in range(start_ind, stop_ind):
      TCfile = root_dir + 'TC_profile_{:0>6}.txt'.format(i)

      #===Data Testing:  Ensure we have no crap data
      #Test Filepath first, if file doesnt exist skip it
      if os.path.isfile(TCfile):
         pass 
      else:
         print("Missing TC File: {:6d}...skipping...".format(i) )
         continue
      
      #Read TC data from ANSYS (has to be cleaned with cleaner.pl)      
      #Note: we skip first line because ANSYS makes funky first lines
      TC_parms = (genfromtxt(TCfile, delimiter=',', skip_header=1))
      
      #===Data Testing:  Ensure we have no crap data
      #Check to make sure there is temp data
      if np.sum(TC_parms) == 0.0:
         print("WARNING: TC DATA = 0.0, No. {:0>6}".format(i))
         continue
      #Check to make sure we cleaned data and dont have any NaNs
      elif np.isnan(np.amax(TC_data)):
         print("WARNING: NAN error: TC DATA, No. {:0>6}".format(i))
         print("Did you clean this data with cleaner.pl...?")
         continue

      # Data in flux array is as follows:
      # [c1, c2, c3, c4]
      #temp1 = [0.0, 0.0, 0.0, 0.0]
      # [S, lambda]
      temp1 = [0.0, 0.0]
      
      # Data in machine specs array:
      # [Bp, P, freq, fx]
      temp2 = [0.0, 0.0, 0.0, 0.0]
      
      file2 = root_dir + 'flux_profile_{:0>6}.txt'.format(i)
      fluxfile = open(file2, 'r')
      for line in fluxfile:
         #Skip header line
         if 'Parameters' in line:
            pass
         #Write data into arrays
#         elif 'c1' in line:
#            temp1[0]= float(line.replace("# c1 = ", " "))
#         elif 'c2' in line:
#            temp1[1]= float(line.replace("# c2 = ", " "))
#         elif 'c3' in line:
#            temp1[2]= float(line.replace("# c3 = ", " "))
#         elif 'c4' in line:
#            temp1[3]= float(line.replace("# c4 = ", " "))
         elif 'S:' in line:
            temp1[0]= float(line.replace("# S:    ", " "))
         elif 'Lambda' in line:
            temp1[1]= float(line.replace("# Lambda [m]:    ", " "))
         elif 'B' in line:
            temp2[0] = float(line.replace("# Bp =  ", " "))
         elif 'P' in line:
            temp2[1] = float(line.replace("# P =   ", " "))
         elif 'R0' in line:
            #Comment the next line for constant freq
            #temp2[2] = -float(line.replace("# R0 time varying, Freq = ", " "))
            temp2[2] = 0
         elif 'fx' in line:   
            temp2[3] = float(line.replace("# fx =    ", " "))

      if temp1[0] == 0.0 or temp1[1] == 0.0: # or temp1[2] == 0.0 or temp1[3] == 0.0:
         print("WARNING: EICH PARAMETER = 0: No. {:0>6}".format(i))
      
      #Build Eich Data numpy array
      for j in range(outN):
         eich_data[i-start_ind][j] = temp1[j]
      
      #Build Machine Specs array
      for j in range(machN):
         # for k in range(3):
            # if j == k:
         mach_data[i-start_ind][j] = temp2[j]
      #mach_data = tf.tanh(mach_data)
      for j in range(int(timeN)):
         TC_data[i-start_ind][j][0] = TC_parms[j][0]
         TC_data[i-start_ind][j][1] = TC_parms[j][1]
         TC_data[i-start_ind][j][2] = TC_parms[j][2]
         TC_data[i-start_ind][j][3] = TC_parms[j][3]
         TC_data[i-start_ind][j][4] = TC_parms[j][4]
   
   
   #===Normalize TC Data
   maxtemp = np.amax(TC_data)
   mintemp = np.amin(TC_data)
   print("\nMaximum Temperature in Dataset: {:f}".format(maxtemp))
   print("Minimum Temperature in Dataset: {:f}".format(mintemp))
   for i in range(0, (stop_ind - start_ind)):
      for j in range(int(timeN)):
         for k in range(int(tcN)):
            val = TC_data[i][j][k]
            TC_data[i][j][k]= 2.0/(maxtemp - mintemp) * (val - mintemp) - 1.0
            
   #===Normalize Machine Spec Data
   for shot in range(0,(stop_ind - start_ind)):
      Bp.append(mach_data[shot][0])
      P.append(mach_data[shot][1])
      freq.append(mach_data[shot][2])
      fx.append(mach_data[shot][3])
   maxBp = np.max(Bp)
   maxP = np.max(P)
   maxfreq = np.max(freq)
   maxfx = np.max(fx)
   minBp = np.min(Bp)
   minP = np.min(P)
   minfreq = np.min(freq)
   minfx = np.min(fx)

   print("Maximum Bp in Dataset: {:f}".format(maxBp))
   print("Minimum Bp in Dataset: {:f}".format(minBp))
   print("Maximum P in Dataset: {:f}".format(maxP))
   print("Minimum P in Dataset: {:f}".format(minP))
   print("Maximum Freq in Dataset: {:f}".format(maxfreq))
   print("Minimum Freq in Dataset: {:f}".format(minfreq))
   print("Maximum fx in Dataset: {:f}".format(maxfx))
   print("Minimum fx in Dataset: {:f}\n".format(minfx))         
   
   #Take Normalized data from range (0,1) to (-1,1)
   for i in range(0, (stop_ind - start_ind)):
      val = mach_data[i][0]
      mach_data[i][0]= 2.0/(maxBp - minBp) * (val - minBp) - 1.0
      val = mach_data[i][1]
      mach_data[i][1]= 2.0/(maxP - minP) * (val - minP) - 1.0
      val = mach_data[i][2]
      #mach_data[i][2]= 2.0/(maxfreq - minfreq) * (val - minfreq) - 1.0
      mach_data[i][2]= 0.0
      val = mach_data[i][3]
      mach_data[i][3]= 2.0/(maxfx - minfx) * (val - minfx) - 1.0
   
   print("Read Data From Files...\n\n")
   
   return TC_data, eich_data, mach_data   # dataset
   


#=======================================================================
#                Functions, Classes, Properties
#=======================================================================

def lazy_property(function):
   attribute = '_' + function.__name__

   @property
   @functools.wraps(function)
   def wrapper(self):
       if not hasattr(self, attribute):
           setattr(self, attribute, function(self))
       return getattr(self, attribute)
   return wrapper

class CNNclass:
   def __init__(self, TCdata, machdata, target):
      self.TCdata = TCdata 
      self.machdata = machdata 
      self.target = target

      self.prediction
      self.error
      self.optimize
      


   @lazy_property
   def prediction(self):
  
      # Convolution 1 - 2X2 filter => fmapN feature maps  
      with tf.name_scope('conv1'):
         W_conv1 = self.weight_variable([5, 5, 1, fmapN])
         b_conv1 = self.bias_variable([fmapN])
         h_conv1 = tf.nn.relu(self.conv2d(self.TCdata, W_conv1) + b_conv1)
      
      # Pooling Layer 1 - Downsample X2
      #with tf.name_scope('pool1'):
      #   h_pool1 = max_pool_2x2(h_conv1)
         
      #Convolution 2 - 2X2 filter => 64 feature maps
      with tf.name_scope('conv2'):
         W_conv2 = self.weight_variable([5, 5, fmapN, 2*fmapN])
         b_conv2 = self.bias_variable([2*fmapN])
      #   h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
         h_conv2 = tf.nn.relu(self.conv2d(h_conv1, W_conv2) + b_conv2)
         
      # Pooling Layer 2 - Downsample X2
      with tf.name_scope('pool2'):
         h_pool2 = self.max_pool_2x2(h_conv2)
      
      # Fully connected layer 0.5
      with tf.name_scope('fchalf'):
         W_fc1 = self.weight_variable([(int(tcN)+1)//2 * int(timeN)//2 *2* fmapN, neuronN])
         b_fc1 = self.bias_variable([neuronN])
         
         h_pool2_flat = tf.reshape(h_pool2, [-1, (int(tcN)+1)//2 * int(timeN)//2 *2* fmapN])
         #h_pool2_flat = tf.reshape(h_conv2, [-1, 3 * 25 * 64])
         h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
         
         
      # # ALTERNATE 1: Fully connected layer 1: for use with no CNN
      # with tf.name_scope('fc1'):
         # W_fc1 = self.weight_variable([int(N), machN])
         # b_fc1 = self.bias_variable([machN])
         
         # h_pool2_flat = tf.reshape(self.TCdata, [-1, int(N)])
         # #h_pool2_flat = tf.reshape(h_conv2, [-1, 3 * 25 * 64])
         # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      
      # # OPTIONAL Fully connected layer 2
      # with tf.name_scope('fc2'):
         # W_fc2 = self.weight_variable([neuronN, machN])
         # b_fc2 = self.bias_variable([machN])
         # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

      # # ALTERNATE 2:  Multiply machine specs with TC conv data
      # # Fully connected layer 3 (MACHINE SPECS ADDED HERE)
      # with tf.name_scope('fc3'):
         # W_fc3 = self.weight_variable([machN, neuronN])
         # b_fc3 = self.bias_variable([neuronN])
         # # Machine specs go are multiplied with h_fc2
         # mach_h = tf.multiply(h_fc1, self.machdata)
         # #mach_h = tf.add(h_fc2, self.machdata)
         # h_fc3 = tf.nn.relu(tf.matmul(mach_h, W_fc3) + b_fc3)
         # #h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
         # #y = tf.matmul(mach_h, W_fc3) + b_fc3
      
      # ALTERNATE 3:  Treat machine specs as separate inputs
      # Fully connected layer 1
      with tf.name_scope('fc4'):
         mach_h = tf.concat([h_fc1, self.machdata], 1)
         
         W_fc4 = self.weight_variable([neuronN+machN, neuronN])
         b_fc4 = self.bias_variable([neuronN])
         h_fc4 = tf.nn.relu(tf.matmul(mach_h, W_fc4) + b_fc4)
         
      # OPTIONAL Fully connected layers 3-8
      with tf.name_scope('fc5'):
         W_fc5 = self.weight_variable([neuronN, neuronN])
         b_fc5 = self.bias_variable([neuronN])
         h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)       
      with tf.name_scope('fc6'):
         W_fc6 = self.weight_variable([neuronN, neuronN])
         b_fc6 = self.bias_variable([neuronN])
         h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)  
      with tf.name_scope('fc7'):
         W_fc7 = self.weight_variable([neuronN, neuronN])
         b_fc7 = self.bias_variable([neuronN])
         h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)       
      with tf.name_scope('fc8'):
         W_fc8 = self.weight_variable([neuronN, neuronN])
         b_fc8 = self.bias_variable([neuronN])
         h_fc8 = tf.nn.relu(tf.matmul(h_fc7, W_fc8) + b_fc8)
      # with tf.name_scope('fc9'):
         # W_fc9 = self.weight_variable([neuronN, neuronN])
         # b_fc9 = self.bias_variable([neuronN])
         # h_fc9 = tf.nn.relu(tf.matmul(h_fc8, W_fc9) + b_fc9)       
      # with tf.name_scope('fc10'):
         # W_fc10 = self.weight_variable([neuronN, neuronN])
         # b_fc10 = self.bias_variable([neuronN])
         # h_fc10 = tf.nn.relu(tf.matmul(h_fc9, W_fc10) + b_fc10)
      # with tf.name_scope('fc6'):
         # W_fc11 = self.weight_variable([neuronN, neuronN])
         # b_fc11 = self.bias_variable([neuronN])
         # h_fc11 = tf.nn.relu(tf.matmul(h_fc10, W_fc11) + b_fc11)  
      # with tf.name_scope('fc7'):
         # W_fc12 = self.weight_variable([neuronN, neuronN])
         # b_fc12 = self.bias_variable([neuronN])
         # h_fc12 = tf.nn.relu(tf.matmul(h_fc11, W_fc12) + b_fc12)       
      # with tf.name_scope('fc8'):
         # W_fc13 = self.weight_variable([neuronN, neuronN])
         # b_fc13 = self.bias_variable([neuronN])
         # h_fc13 = tf.nn.relu(tf.matmul(h_fc12, W_fc13) + b_fc13)
      # with tf.name_scope('fc9'):
         # W_fc14 = self.weight_variable([neuronN, neuronN])
         # b_fc14 = self.bias_variable([neuronN])
         # h_fc14 = tf.nn.relu(tf.matmul(h_fc13, W_fc14) + b_fc14)       
      # with tf.name_scope('fc10'):
         # W_fc15 = self.weight_variable([neuronN, neuronN])
         # b_fc15 = self.bias_variable([neuronN])
         # h_fc15 = tf.nn.relu(tf.matmul(h_fc14, W_fc15) + b_fc15)
         
      
      # Fully connected layer 5 - Output Layer
      with tf.name_scope('fc_OUT'):
         W_fc_out = self.weight_variable([neuronN, int(outN)])
         b_fc_out = self.bias_variable([int(outN)])
         y = tf.matmul(h_fc8, W_fc_out) + b_fc_out

      return y
      

      
   @lazy_property
   def loss(self):
      #loss = tf.losses.mean_squared_error(self.prediction, self.target)
      
#      loss = tf.reduce_mean( tf.matmul(tf.abs(self.prediction - self.target), slam_range) )
      loss = 1000*tf.reduce_mean(tf.abs(self.prediction - self.target)) 

      return loss
   @lazy_property
   def optimize(self):
      optimizer = tf.train.AdamOptimizer(lr)
      # op1 = optimize.minimize(self.loss1)
      # op2 = optimize.minimize(self.loss2)
      # op3 = optimize.minimize(self.loss3)
      # op4 = optimize.minimize(self.loss4)
      #optimizer = tf.train.GradientDescentOptimizer(lr)
      return optimizer.minimize(self.loss)

   # @lazy_property
   # def optimize(self):
      # #optimizer = tf.train.AdamOptimizer(lr)
      # # op1 = optimize.minimize(self.loss1)
      # optimizer = tf.train.GradientDescentOptimizer(lr)
      # return optimizer.minimize(self.loss)
   # @lazy_property
   # def optimize(self):
      # #optimizer = tf.train.AdamOptimizer(lr)
      # # op2 = optimize.minimize(self.loss2)
      # optimizer = tf.train.GradientDescentOptimizer(lr)
      # return optimizer.minimize(self.loss)
   # @lazy_property
   # def optimize(self):
      # #optimizer = tf.train.AdamOptimizer(lr)
      # # op3 = optimize.minimize(self.loss3)
      # optimizer = tf.train.GradientDescentOptimizer(lr)
      # return optimizer.minimize(self.loss)
   # @lazy_property
   # def optimize(self):
      # #optimizer = tf.train.AdamOptimizer(lr)
      # # op4 = optimize.minimize(self.loss4)
      # optimizer = tf.train.GradientDescentOptimizer(lr)
      # return optimizer.minimize(self.loss)
   # @lazy_property
   # def optimize(self):
      # optimizer = tf.train.AdamOptimizer(lr)
      # #optimizer = tf.train.GradientDescentOptimizer(lr)
      # return optimizer.minimize(self.loss)

   @lazy_property
   def error(self):
      error = tf.subtract(self.target, self.prediction)
#      error = tf.matmul((self.prediction - self.target), C_range)
      return error

   @staticmethod   
   # conv2d returns a 2d convolution layer with full stride   
   def conv2d(x, W):
     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

   @staticmethod
   # max_pool_2x2 downsamples a feature map by 2X
   def max_pool_2x2(x):
     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
   
   @staticmethod
   # Generates a weight variable of a given shape
   def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.01)
      return tf.Variable(initial)
   @staticmethod
   # Generates a bias variable of a given shape
   def bias_variable(shape):
      initial = tf.constant(0.0, shape=shape)
      return tf.Variable(initial)

#=======================================================================
#                Main Program
#=======================================================================


def main():
   train_statcounter = 0.0
   train_hits = 0.0
   train_acc_window = np.zeros((int(box_window_size)))
   test_statcounter = 0.0
   test_hits = 0.0
   test_acc_window = np.zeros((int(box_window_size)))

   
   
   #=== First, create training Dataset
   TC_data, eich_data, mach_data = import_data(experfirst, experN + experfirst)
   # Organize Data into dataset object for tensorflow
   dataset = tf.data.Dataset.from_tensor_slices((TC_data, mach_data, eich_data))
# Commented for triplet.  All other cases uncomment
   dataset = dataset.shuffle(buffer_size=10000)
   dataset = dataset.batch(batch_size)
   dataset = dataset.repeat(num_epochs)
   # Create iterator for dataset
   iterator = dataset.make_one_shot_iterator()
   next_element = iterator.get_next()
   
   #=== Create Test Dataset
   TC_data_test, eich_data_test, mach_data_test = import_data(experN + experfirst, experN + experfirst + testN )
   # Organize Data into dataset object for tensorflow
   dataset_test = tf.data.Dataset.from_tensor_slices((TC_data_test, mach_data_test, eich_data_test))
   dataset_test = dataset_test.shuffle(buffer_size=10000)
   dataset_test = dataset_test.batch(1)
   dataset_test = dataset_test.repeat(num_epochs)
   # Create iterator for dataset
   iterator_test = dataset.make_one_shot_iterator()
   next_element_test = iterator.get_next()
   
   # Input Data
   with tf.name_scope('TC_input_data'):
      data_TC = tf.placeholder(tf.float32, [None, int(timeN), int(tcN), 1])
   
   with tf.name_scope('mach_input_data'):
      data_mach = tf.placeholder(tf.float32, [None, machN])
   
   # Expected Result (y_ is expected)
   with tf.name_scope('Expected_Result'):
      target = tf.placeholder(tf.float32, [None, outN])

   # Build the model
   model = CNNclass(data_TC, data_mach, target)
   
   sess = tf.InteractiveSession()
   
   # Add ops to save and restore all the variables.
   saver = tf.train.Saver()
   
   # If weight importer flag is set then we import weights and dont
   # initialize variables
   if import_flag ==1:
      global weight_path
      saver = tf.train.import_meta_graph(meta_path)
      saver.restore(sess, weight_path)
      print("Read model from file.  Model Restored\n")
   else:
      tf.global_variables_initializer().run()

   # Create Filewriter for Tensorboard Visualization
   # writer = tf.summary.FileWriter("/tmp/tf_test")
   # writer.add_graph(sess.graph)
   
   #Debugging stuff
   #var = [v for v in tf.trainable_variables() if v.name == "fc1/Variable_1:0"][0]
   
   #====================================================================
   #             Training
   #====================================================================   
   # Train
   for epoch in range(num_epochs):
      #Training Data
      x_TC, x_mach, y_eich = sess.run(next_element)
      sess.run(model.optimize, {data_TC: x_TC, data_mach: x_mach, target: y_eich})
      error = sess.run(model.error, {data_TC: x_TC, data_mach: x_mach, target: y_eich})

      #Test Data
      x_TC_test, x_mach_test, y_eich_test = sess.run(next_element_test)
      test_error = sess.run(model.error, {data_TC: x_TC_test, data_mach: x_mach_test, target: y_eich_test})
      
      #===== Accuracy Windows =====
      # Training Batch Statistics
      for idx in range(len(error)):
         train_statcounter += 0.02
         for err_idx in range(outN):
            # Check if prediction was within allowerable tolerance
            if err_idx == 0:
               if abs(error[idx][err_idx]) < error_thresh_S:
                  train_hits += 1.0
            elif err_idx == 1:
               if abs(error[idx][err_idx]) < error_thresh_lam:
                  train_hits += 1.0               
      # Calculate Accuracy
      # Shift boxcar window then write new value in: moving window
      train_acc_window = np.roll(train_acc_window, 1)
      train_acc_window[0] = train_hits/train_statcounter
      # Reset accuracy counters
      train_hits = 0.0
      train_statcounter = 0.0

      # Test Batch Statistics
      for idx in range(len(test_error)):
         test_statcounter += 0.02
         for err_idx in range(outN):
            # Check if prediction was within allowerable tolerance
 #           print("test error======={:f}".format(test_error[0][0]))
            if err_idx == 0:
               if abs(test_error[idx][err_idx]) < error_thresh_S:
                  test_hits += 1.0
            elif err_idx == 1:
               if abs(test_error[idx][err_idx]) < error_thresh_lam:
                  test_hits += 1.0 

      # Calculate Accuracy
      # Shift boxcar window then write new value in: moving window
      test_acc_window = np.roll(test_acc_window, 1)
      test_acc_window[0] = test_hits/test_statcounter
      # Reset accuracy counters
      test_hits = 0.0
      test_statcounter = 0.0
      
      if epoch % samplelen == 0:
         loss_value = sess.run(model.loss, {data_TC: x_TC, data_mach: x_mach, target: y_eich} )
         
         #Append for plot 
         train_loss_results.append(loss_value)
         train_loss_epoch.append(epoch)

         error_sum = np.zeros((2))
         # Batch Statistics
         for batch_idx in range(batch_size): 
            for err_idx in range(2):
               # Regular Sum
               error_sum[err_idx] += abs(error[batch_idx][err_idx])
         
         #Append for plot 
         for err_idx in range(2):
            train_error[epoch//samplelen][err_idx] = error_sum[err_idx]/batch_size
         
         #Accuracy Stats
         train_accuracy = sum(train_acc_window)/box_window_size
         train_acc.append(train_accuracy)
         
         test_accuracy = sum(test_acc_window)/box_window_size
         test_acc.append(test_accuracy)
         

         #=== Test Dataset   
         # loss_test = sess.run(model.loss, {data_TC: x_TC_test, data_mach: x_mach_test, target: y_eich_test})   
         # test_loss_results.append(loss_test)
         # for err_idx in range(4):
            # test_error[epoch//samplelen][err_idx] = error_test[0][err_idx]

      
      if epoch % printlen == 0:
         test_resulty_ = sess.run(model.target, {data_TC: x_TC_test, data_mach: x_mach_test, target: y_eich_test})
         test_resulty = sess.run(model.prediction, {data_TC: x_TC_test, data_mach: x_mach_test, target: y_eich_test})
         # resulty_ = sess.run(model.target, {data_TC: x_TC, data_mach: x_mach, target: y_eich})
         # resulty = sess.run(model.prediction, {data_TC: x_TC, data_mach: x_mach, target: y_eich})
         
         # Debugging Stuff
         #print(tf.trainable_variables())
         #test = sess.run(var)
         #print(test)
         
         
         print("==========================Epoch {:06d}:=========================".format(epoch,))
         print("Time Elapsed ~ {:d} seconds\n".format(int(time.time() - start_time)),)
         print("Batch Averaged Training Data Error:")  
         print(train_error[epoch//samplelen])
         print("Training Data Loss:")  
         print(loss_value)
         print("Training Accuracy:")  
         print(train_accuracy)
         #print("\n")

         print("\n+++=== Test Data Example Results ===+++")
         print("Expected: ")
         print(test_resulty_[0])
         print("Predicted: ")
         print(test_resulty[0])
         print("\n") 
         print("Test Data Accuracy:")  
         print(test_accuracy)      
         print("\n")   
         
         
         # print("\n+++=== Test Data Results ===+++")
         # print("Expected: ")
         # print(test_resulty_[0])
         # print("Predicted: ")
         # print(test_resulty[0])
         # print("\n")
      if test_accuracy > acc_thresh:
         break
  
   #====================================================================
   #             Save Weights for Predictor
   #====================================================================

   #Save Weights
   ts = time.time()
   st = dt.datetime.fromtimestamp(ts).strftime('%Y%m%d--%H_%M_%S')
   os.mkdir(os.path.join(weights_dir, st))
   weight_path = os.path.join(weights_dir, st, 'weights' )
   print("\n Saving weights to:")
   print(weight_path)
   save_path = saver.save(sess, weight_path)
  
   #====================================================================
   #             Plots
   #====================================================================
   
   # Plot Loss with Matplotlib
   plt.figure(1)
   
   #plt.subplot(211)
   plt.title("CNN: LR = {:.4f}; BatchSize = {:4d}; Neuron N = {:4d}; FMaps: {:4d}".format(lr, batch_size, neuronN, fmapN))
   #plt.title('Loss vs. Epoch')
   axes = plt.gca()
   axes.set_ylim([0.0,100.0])
   plt.plot(train_loss_epoch, train_acc, 'b', label="Training Accuracy")
   plt.plot(train_loss_epoch, test_acc, 'g', label="Test Accuracy")
   #plt.plot(train_loss_epoch, train_loss_results, 'b', label="Training Data")
   #plt.plot(train_loss_epoch, test_loss_results, 'g', label="Test Data")
   #plt.xlabel('Epoch')
   plt.ylabel('Accuracy (within tolerance)')
   plt.xlabel('Epoch')
   plt.legend(loc='upper right')
   # ts = time.time()
   # st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d--%H:%M:%S')
   # plotname = "lr{:f}_bs{:d}_neuronN{:d}_fmap{:d}.png".format(lr, batch_size, neuronN, fmapN)
   # plt.savefig(os.path.join(figure_dir, st ,plotname)) # , bbox_inches='tight'
   plt.show()
   
   #===LOSS PLOT
   # plt.figure(2)
   # #plt.subplot(212)
   # axes = plt.gca()
   # axes.set_ylim([0.0,0.01])
   # #Train Data
   # plt.plot(train_loss_epoch, [row[0] for row in train_error], label="Train S Error")
   # plt.plot(train_loss_epoch, [row[1] for row in train_error], label="Train Lambda Error")
   # #plt.plot(train_loss_epoch, [row[2] for row in train_error], label="Train C3 Error")
   # #plt.plot(train_loss_epoch, [row[3] for row in train_error], label="Train C4 Error")
   
   # #Test Data
   # # plt.plot(train_loss_epoch, [row[0] for row in test_error], label="Test C1 Error")
   # # plt.plot(train_loss_epoch, [row[1] for row in test_error], label="Test C2 Error")
   # # plt.plot(train_loss_epoch, [row[2] for row in test_error], label="Test C3 Error")
   # # plt.plot(train_loss_epoch, [row[3] for row in test_error], label="Test C4 Error")
   # plt.legend(loc='upper right')
   # plt.xlabel('Epoch')
   # plt.ylabel('Absolute Error')
   # #ts = time.time()
   # #st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d--%H:%M:%S')
   # #plotname = "lr{:f}_bs{:d}_neuronN{:d}_fmap{:d}.png".format(lr, batch_size, neuronN, fmapN)
   # #plt.savefig(os.path.join(figure_dir, st ,plotname)) # , bbox_inches='tight'
   # plt.show()
   


   

if __name__ == '__main__':
   main()




















   







