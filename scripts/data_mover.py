# cnn_20180721.py

# Date:         20180728
# Description:  Separates C Value "triplets" into different directories
# Engineer:     Tom Looby

import os
import shutil
import numpy as np


#===USE THESE FOR TRAINING DATASETS
#Root directory where we are working
new_dir = '/home/workhorse/school/grad/masters/tensorflow/data_nosweep_all/'
root_dir1_base = '/home/workhorse/school/grad/masters/tensorflow/data_20s_nosweep_triple/'
root_dir2_base = '/home/workhorse/school/grad/masters/tensorflow/data_20s_nosweep_allrandom/'


#==== EDIT THESE ====
#Index TC Data Starts at
start_idx1 = 1
#Index TC Data stops at
stop_idx1 = 5106


#Index TC Data Starts at
start_idx2 = 1
#Index TC Data stops at
stop_idx2 = 2938


# This is number we start relabeling at
trip_idx = 1
#==== EDIT THESE ====



#===USE THESE FOR PREDICTOR
#Root directory where we are working
# root_dir_base = '/home/workhorse/school/grad/masters/tensorflow/data_20s_Cs_const_nosweep/'
# root_dir1 = '/home/workhorse/school/grad/masters/tensorflow/data_20s_Cs_const_nosweep/01/'
# root_dir2 = '/home/workhorse/school/grad/masters/tensorflow/data_20s_Cs_const_nosweep/02/'
# root_dir3 = '/home/workhorse/school/grad/masters/tensorflow/data_20s_Cs_const_nosweep/03/'

# #==== EDIT THESE ====
# #Index TC Data Starts at
# start_idx = 1
# #Index TC Data stops at
# stop_idx = 300
# # This is number we start relabeling at
# trip_idx = 1
# #==== EDIT THESE ====



print("=== Data Mover ===\n")
for idx in range(start_idx1,stop_idx1+1):

   #Copy TC Data
   oldpathTC = root_dir1_base + 'TC_profile_{:0>6}.txt'.format(idx)
   newpathTC = new_dir + 'TC_profile_{:0>6}.txt'.format(trip_idx)
   shutil.copy2(oldpathTC,newpathTC)
   
   #Copy Flux Data
   oldpathflux = root_dir1_base + 'flux_profile_{:0>6}.txt'.format(idx)
   newpathflux = new_dir + 'flux_profile_{:0>6}.txt'.format(trip_idx)
   shutil.copy2(oldpathflux,newpathflux)
   

   #Increment Triple Counter
   trip_idx += 1



for idx in range(start_idx2,stop_idx2+1):

   #Copy TC Data
   oldpathTC = root_dir2_base + 'TC_profile_{:0>6}.txt'.format(idx)
   newpathTC = new_dir + 'TC_profile_{:0>6}.txt'.format(trip_idx)
   shutil.copy2(oldpathTC,newpathTC)
   
   #Copy Flux Data
   oldpathflux = root_dir2_base + 'flux_profile_{:0>6}.txt'.format(idx)
   newpathflux = new_dir + 'flux_profile_{:0>6}.txt'.format(trip_idx)
   shutil.copy2(oldpathflux,newpathflux)
   

   #Increment Triple Counter
   trip_idx += 1

   
   
#   pint("\nOLDPATH: ")
#   print(oldpathTC)
#   print("NEWPATH: ")
#   print(newpathTC)

print("\nScript Completed...Data Moved\n\n")
      
