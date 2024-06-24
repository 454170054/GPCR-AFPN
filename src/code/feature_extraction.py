from gensim.models import word2vec, FastText
import numpy as np


def get_text(seqs, name, dir):
    file_name = name + '_' + "{}.txt".format(2)
    file_path = "../../resources/" + dir + "/" + file_name
    with open(file_path, "w") as fr:
        for j in seqs:
            text = ""
            for k in range(len(j) - 1):
                if k != len(j) - 2:
                    text = text + j[k: k + 2] + " "
                else:
                    text = text + j[k: k + 2]
            fr.write(text + "\n")
    return file_path


def train_model(seqs, dir):
    '''
    train the fastText models
    '''
    name = 'train'
    file_path = get_text(seqs, name, dir)
    sentences = word2vec.LineSentence(file_path)
    model2 = FastText(sentences, min_count=0, vector_size=288, window=7, epochs=30, workers=8)
    return model2


def fastText_features(file_path, model):
    '''
    extract the features
    '''
    f = open(file_path, 'r')
    content = f.readlines()
    f.close()
    count = 0
    features = None
    for line in content:
        line = line.split()
        single_feature = np.mean(model.wv[line], axis=0, keepdims=True)
        if count == 0:
            features = single_feature
            count += 1
        else:
            features = np.r_[features, single_feature]
    return features


def get_features_from_datasets(seqs, name, model, dir):
    '''
    concatenate the three kinds features
    '''

    if name == 'train':
        pass
    else:
        get_text(seqs, name, dir)
    features = fastText_features(r'../../resources/{}/{}_2.txt'.format(dir,name), model)
    return features