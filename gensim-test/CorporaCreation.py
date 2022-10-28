import numpy as np

from numpy.random import default_rng


rng = default_rng()


topics = 2 # K
words = 100 # V
num_docs = 3
words_per_doc = 30

alpha = np.full(topics, 0.1) # K-dimensional
beta = np.full(words, 0.1) # V-dimensional

words_in_topic_distrs = []
for i in range(topics):
    distr = rng.dirichlet(beta)
    words_in_topic_distrs.append(distr)


corpus = []
for i in range(num_docs):
    topic_distr = rng.dirichlet(alpha)
    doc = []
    # creating the document
    for j in range(words_per_doc):
        # determine a topic for word
        topic_draw =rng.multinomial(1,topic_distr)
        topic = np.where(topic_draw==1)[0][0]
        print(topic)
        word_draw = rng.multinomial(1,words_in_topic_distrs[topic]) # determine a word of the topic
        word = np.where(word_draw==1)[0][0]
        doc.append(word)

    corpus.append(doc)
