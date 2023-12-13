# GRAPES on Link Prediction
Thie repo adopts the orignal GRAPES framework (https://github.com/dfdazac/grapes/tree/main) to link prediction tasks. 

## Instructions
1. Install dependencies
Create a conda environment with the provided file, then activate it:
  ```
  conda env create -f environment.yml
  conda activate grapes
```
2. Train a model
Run the following to train a GCN classifier on the Cora dataset:
```
  python main.py
```
Be sure to change the Arguments Class in main.py to match CiteSeer and Cora config files based on the dataset being used. 
