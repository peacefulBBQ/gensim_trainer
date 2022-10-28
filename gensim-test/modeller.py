import numpy as np
import re
import matplotlib.pyplot as plt
from gensim.models import LdaModel
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






def preprocessing(corpus):
    stop_words = stopwords.words('english')
    clean_docs = []

    for doc in corpus:
        text = doc.splitlines()
        text = text[text.index("")+1:]

        done = gensim.utils.simple_preprocess(" ".join(text))

        without_stops = [word for word in done if word not in stop_words]
        clean_docs.append(without_stops)

    # Creating a Dictionary for the Corpus
    my_dict = corpora.Dictionary(clean_docs)

    # Turn corpus into Bag of Words
    BoW_corpus = [my_dict.doc2bow(doc) for doc in clean_docs]


    return my_dict, BoW_corpus



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




