# python_tester.py


from __future__ import absolute_import, division, print_function
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from numpy import genfromtxt
from numpy import unravel_index
from numpy.linalg import inv
import functools
import time
import datetime as dt
#import tensorflow.contrib.eager as tfe
#tf.enable_eager_execution()
import csv   
   
   
#====================================================================
#             Newton's Method (solve nonlinear Eich system)
#====================================================================    
print("\nSolving Nonlinear System with Newton's Method...")
counter = 0
# Initial Guess for C
c = ([1.323], [0.158], [-1.1])
c=np.array(c)
dF = np.zeros((3,3))
F = np.zeros((3,1))
mach = np.zeros((3,2))
lam = np.zeros((3,1))
newt_thresh = 1e-9
newt_error = 1.0

C2 = 1.5
C3 = -0.1
C4 = -1.0

mach[0][0] = 0.3633
mach[0][1] = 1.983
mach[1][0] = 0.256
mach[1][1] = 3.482
mach[2][0] = 0.508
mach[2][1] = 3.032

lam[0][0] = C2 * (mach[0][1]**C3) * (mach[0][0]**C4)
lam[1][0] = C2 * (mach[1][1]**C3) * (mach[1][0]**C4)
lam[2][0] = C2 * (mach[2][1]**C3) * (mach[2][0]**C4)


while True:
   counter += 1
   for shot in range(3):
      F[shot] = (c[0] \
                  # P^C3
                  * np.power(mach[shot][1], c[1]) \
                  # B^C4
                  * np.power(mach[shot][0], c[2]) \
                  # lambda
                  - lam[shot][0])
      for cval in range(3):        
         if cval == 0:
            dF[shot][cval] = np.power(mach[shot][1], c[1]) * np.power(mach[shot][0], c[2]) # P^C3 * B^C4
         
         elif cval == 1:
            dF[shot][cval] = (c[0] \
                              * np.power(mach[shot][1], c[1]) \
                              * np.power(mach[shot][0], c[2]) \
                             # ln(P)
                              * np.log(mach[shot][1]))

         elif cval == 2:
            dF[shot][cval] = (c[0] \
                              * np.power(mach[shot][1], c[1]) \
                              * np.power(mach[shot][0], c[2]) \
                              # ln(B)
                              * np.log(mach[shot][0]))
                             
   
   c_new = c - np.matmul(inv(dF), F)
   newt_error = (np.sum( np.abs(c_new - c), axis=0))
   c = c_new
   if (newt_error < newt_thresh):
      break
   print("\n============================")
   print(c)
print("Num Iter:")
print(counter)
print("C Values")
print(c)
