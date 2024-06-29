from gensim.models import FastText
import pandas as pd
import numpy as np
from src.code.network import create_model
from src.code.feature_extraction import get_text, fastText_features
import os
import glob
import re


def union():
    files = glob.glob("../../resources/model/fastText/*.bin")
    files.sort(key=lambda l: int(re.findall('\d+', l)[0]))
    with open("../../resources/model/fastText/fastText.model.wv.vectors_ngrams.npy", "wb") as writer:
        for file in files:
            with open(file, "rb") as reader:
                writer.write(reader.read())


class Predictor:

    def __init__(self) -> None:
        super().__init__()
        self.dir = "predict"

    def predict(self, file_path, job_id):
        df = pd.read_csv(file_path, sep=',', header=0)
        seqs = df['seq'].values.tolist()
        file_path = get_text(seqs, job_id, self.dir)
        if not os.path.exists("../../resources/model/fastText/fastText.model.wv.vectors_ngrams.npy"):
            union()
        fast_text_model = FastText.load("../../resources/model/fastText/fastText.model")
        features = fastText_features(file_path, fast_text_model)
        print(features)
        print(features.shape)
        model = create_model(features.shape)
        model.load_weights("../../resources/model/model.h5")
        p = model.predict(features)[2]
        prediction = np.zeros_like(p)
        prediction[p >= 0.5] = 1
        df['probability'] = p
        df['Label'] = prediction.astype(int)
        print(df)
        df.to_csv(f'../../resources/predict/results/{job_id}.csv')

