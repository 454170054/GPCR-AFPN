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
3. unzip the rar file "resources/model/fastText.model.wv.vectors_ngrams.rar" to its parent dir "model".

# Using GPCR-AFPN<hr>

* Using GPCR-AFPN to identify GPCRs is easy. You can see the examples in
  this <a href='https://github.com/454170054/GPCR-AFPN/blob/main/src/demo/predict_demo.ipynb'>notebook</a>.<br>
* If you want to repeat the experiment results reported in the paper, you can execute the
  files <a href='https://github.com/454170054/GPCR-AFPN/blob/main/src/code/cross_validation.py'>
  cross_validation.py</a> and <a href='https://github.com/454170054/GPCR-AFPN/blob/main/src/code/independent_test.py'>
  independent_test.py</a>.
* If you want to use different datasets to train a predictor, you can see the
  file <a href='https://github.com/454170054/GPCR-AFPN/blob/main/src/demo/train_model_demo.ipynb'>
  train_model_demo.ipynb</a>.


