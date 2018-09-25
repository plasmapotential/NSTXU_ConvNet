# NSTX-U / UTK TENSORFLOW Directory
## Contains tensorflow scripts for CNNs, RNNs, etc.

All scripts are located in the ./scripts directory.  The latest revision utilizes a CNN that predicts S and lambda.  

To run the latest scripts that are in the ./scripts directory, you will need the weights for the CNN matrix.
These weights are not included here, and must be created by running the 'training' script before the 
'prediction' scripts.

There may be some path references that you need to change, but I always define paths at the top of the code to make it easy.

There are many previous revision that may or may not work included in the ./scripts/old_revs directory.
Additionally, in the scripts directory there is a cleaner script, a data mover script, a newtons method script,
and a couple others.  To train a model, run the SLam_predictor.py script.  To test a model, run the 
SLam_tester.py script.


