from gensim.models import LdaModel
import os
import math

from sklearn.datasets import fetch_20newsgroups
import modeller

# my_model = LdaModel.load("model_file")

# print(my_model.top_topics())
newsgroups_train = fetch_20newsgroups(subset='train')
corpus = newsgroups_train.data


my_dict, corpus = modeller.preprocessing(corpus)


tenth = math.floor(len(corpus)/10)
for i in range(10):
    corp_size = tenth * (i+1)
    print(i)
    name = str(corp_size)
    if not os.path.exists("model_" + name):
        os.makedirs("model_" + name)
        os.chdir("./model_" + name)
        modeller.build_model(my_dict, corpus[:corp_size], 50)
        os.chdir("..")
    