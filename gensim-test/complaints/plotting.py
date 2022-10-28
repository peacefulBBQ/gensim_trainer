 
from sklearn.datasets import fetch_20newsgroups
import re
import matplotlib.pyplot as plt
import os
from gensim import models
import numpy as np
import modeller


def plotting(dir_name):
    p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
    matches = [p.findall(l) for l in open(dir_name + 'gensim.log')]
    matches = [m for m in matches if len(m) > 0]
    print(matches)
    tuples = [t[0] for t in matches]
    perplexity = [float(t[1]) for t in tuples]
    likelihood = [float(t[0]) for t in tuples]
    iter = list(range(0,len(tuples)*10,10))
    plt.plot(iter,likelihood,c="black")
    plt.ylabel("log likelihood")
    plt.xlabel("iteration")
    plt.title("Topic Model Convergence")
    plt.grid()
    plt.savefig(dir_name + "convergence.pdf")
    plt.close()




def getting_perplexities():
    _, BoW_corpus = modeller.preprocessing(fetch_20newsgroups(subset='train').data)
    
    
    values = [] # list of all document, perplexity pairs
    for x in os.listdir("."):
        if x.startswith("model"):
            print(x)
            doc_count = int(x.split("_")[1]) # get the number of documents the model was trained on
            os.chdir(x)
            model = models.ldamodel.LdaModel.load("model_file")

            bound = model.log_perplexity(BoW_corpus)
            # perplexity = np.exp2(-bound)

            values.append([doc_count, bound])
            # print(model.top_topics(BoW_corpus)[0])
            os.chdir("..")

    values.sort(key=lambda x : x[0])
    print(values)

    fig, ax = plt.subplots()

    plt.plot([val[0] for val in values], [val[1] for val in values], xlabel= "Number of documents", ylabel= "bound")
    plt.axes.Axes.set_xlabel("Number of documents")
    plt.axes.Axes.set_ylabel("bound")
    plt.show()
    



getting_perplexities()
