import gensim
from gensim.models import TfidfModel, LdaModel
from gensim import corpora
import numpy as np
import matplotlib.pyplot as plt
from DataModel import DataModel
import os
import logging

class MyModel:
    def __init__(self, corpus, directory):
        self.corpus = corpus
        self.dict = corpora.Dictionary(corpus)
        self.directory = directory

        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + "/plots")

        self.tfidfs = None
    

    def get_tfidfvals(self):
        # nach Sek Hansen
        if self.tfidfs != None:
            return self.tfidfs
        
        tfidfs = []

        for i in self.dict.cfs.keys():
            tf = 1 + np.around(np.log2(self.dict.cfs[i]), 4)
            df = np.around(np.log2(self.dict.num_docs / self.dict.dfs[i]), 4)
            tfidfs.append(np.around(tf*df, 2))
        
        self.tfidfs = np.array(tfidfs)

        return self.tfidfs

    def make_tfidf_plot(self):
        word_nums = np.arange(len(self.dict.cfs.keys()))
        sorted = -np.sort(-self.tfidfs) # sort in descending order
        plt.plot(word_nums, sorted,c="black")
        plt.xlabel("Ranks of terms by TF-IDF")
        plt.ylabel("TF-IDF weight")
        plt.title("TF-IDF Analysis for " + str(self.dict.num_docs) + " documents")
        plt.savefig(self.directory + "/plots/TF-IDF_" + str(self.dict.num_docs))
        plt.close()


    def drop_tfidf(self, thresh):
        drop_words = [self.dict[x] for x in self.dict.cfs.keys() if self.tfidfs[x] <= thresh]
        clean_docs = [[word for word in doc if doc not in drop_words] for doc in self.corpus]
        
        # remove empty documents
        clean_docs = [doc for doc in clean_docs if doc]
        self.dict = corpora.Dictionary(clean_docs)
        self.corpus = clean_docs
        return clean_docs


    def build_LDA(self, topics, passes, chunks=1000, update=1):
        logging.basicConfig(filename=self.directory+"/gensim.log", 
                            format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        
        corpus = [self.dict.doc2bow(doc) for doc in self.corpus]
        cut_at = int(np.around((3/4) * len(corpus), 0))
        training = corpus[:cut_at]
        self.testing = corpus[cut_at+1:]
        lda_model = LdaModel(
            corpus = training,
            id2word=self.dict,
            chunksize=chunks,

            alpha='auto', # document-topic distribution
            eta='auto',   # topic-word distribution

            num_topics= topics,
            passes= passes,
            eval_every= 200,
            update_every= update
        )
        if not os.path.exists(self.directory+"/model"):
            os.makedirs(self.directory+"/model")
        lda_model.save(self.directory+"/model/" +str(topics)+"tops")
        self.lda = lda_model
       
    

    def get_avg_doc_len(self):
        return sum([len(doc) for doc in self.corpus]) / len(self.corpus)
    
    def get_doc_num(self):
        return len(self.corpus)

    def get_perplexity(self):
        return self.lda.log_perplexity(self.testing)





'''
root = "gensim-test/complaints"
data_model = DataModel(root+"/data/raw_data.json")
corpus = data_model.clean_data()

model = MyModel(corpus, root+"/models"+"/testing")
model.get_tfidfvals()
model.make_tfidf_plot()

'''
