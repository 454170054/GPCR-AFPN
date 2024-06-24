# GPCR-AFPN
<hr>
GPCR-AFPN is an advanced method for indentifying GPCRs. This method leverages the FastText algorithm to effectively extract 
features from protein sequences. Additionally, it employs a powerful deep neural network as the predictive model, to 
improve prediction accuracy.

# Build the conda environment<hr>
Before using GPCR-AFPN, you are recommended to create a new conda virtual environment.
1. clone the repository.
2. create a new conda environment. You can execute the following commands: 
   - conda create -n gpcr-afpn python=3.10.12 
   - conda activate gpcr-afpn
   - cd gpcr-afpn
   - pip install -r requirements.txt


# Using GPCR-AFPN<hr>
* Using GPCR-AFPN to identify GPCRs is easy. You can see the examples in this <a href='https://github.com/454170054/iSUMO-RsFPN/blob/main/code/predict/demo.ipynb'>notebook</a>.<br>
* If you want to repeat the experiment results reported in the paper, you can execute the file <a href='https://github.com/454170054/iSUMO-RsFPN/blob/main/code/experimental_result/isumo-rsfpn.ipynb'>isumo-rsfpn.ipynb</a>.
* If you want to use different datasets to train a predictor, you can see the file <a href='https://github.com/454170054/iSUMO-RsFPN/blob/main/code/predictor/train_predictor.py'>train_predictor.py</a>.


