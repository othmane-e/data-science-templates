#!/usr/bin/env bash -l
if [[ "$#" -ge 1 ]]
then
    #set -x
    conda create -n $1
    conda init bash
    conda activate $1
    conda install pip numpy pandas matplotlib seaborn tqdm scikit-learn xgboost
    conda install -c anaconda ipykernel
    ipython kernel install --user --name=$1
    conda deactivate
else
  echo "Error: please enter a name for the environment."
fi



