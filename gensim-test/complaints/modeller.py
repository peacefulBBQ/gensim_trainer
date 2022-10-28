import numpy as np
import re
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.models import TfidfModel
from sklearn.datasets import fetch_20newsgroups
import nltk
import gensim
from gensim import corpora, models
import functools as f
import tempfile
nltk.download("stopwords")
from nltk.corpus import stopwords
from gensim.test.utils import datapath
import logging



# BoW_corpus = [my_dict.doc2bow(doc) for doc in clean_docs]


# Train LDA model.
def build_model(my_dict, BoW_corpus, topic_num):
    logging.basicConfig(filename="gensim.log", format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = LdaModel(
        corpus=BoW_corpus,
        id2word=my_dict,
        chunksize=1000,

        alpha='auto', # document-topic distribution
        eta='auto',   # topic-word distribution

        iterations=10,
        num_topics= topic_num,
        passes=50,
        eval_every= 100,
        update_every= 1
    )
    model.save("./model_file")



#bound = model.log_perplexity([BoW_corpus[-1]])
#perplexity = np.exp2(-bound)
#print(perplexity)
#print(model.top_topics(BoW_corpus)[0])




