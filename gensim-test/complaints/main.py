from gensim.models import TfidfModel
from gensim.models import LdaModel
import modeller
import random
from DataModel import DataModel
from MyModel import MyModel
import matplotlib.pyplot as plt
import numpy as np

# make decision depending to drop words depending on tfidf

def plot_perplexities(topic_lst, perplexities, doc_nums):
    # figure, axis = plt.subplots(len(perplexities))
    for idx,perplexity in enumerate(perplexities): 
        plot = plt.subplot2grid((4, 0), (idx,0), colspan=1, rowspan=1)
        plot.plot(topic_lst, np.around(-np.array(perplexity),2))
        plot.set_title("Perplexity for "+ str(doc_nums[idx]) + " documents")
        plot.set_xlabel("# topics")
        plot.set_ylabel("Log perplexity")

    plt.tight_layout()
    plt.savefig("Overview")
    plt.close()


root = "gensim-test/complaints"
data_model = DataModel(root+"/data/raw_data.json")
docs = data_model.clean_data(2100)
random.shuffle(docs)
corpus = docs[:2000]
test_set = docs[2000:]
models = []

perplexities = []
topic_lst = [10,20,30]
doc_nums = [250,500,1000,2000] 
for idx, doc_num in enumerate(doc_nums):
    perplexities.append([])
    print(doc_num)
    root = "gensim-test/complaints"
    model = MyModel(corpus[:doc_num], root+"/models"+"/model_"+str(doc_num))
    model.get_tfidfvals()
    # model.make_tfidf_plot() # use this only when you still need to find the thresholds

    model.drop_tfidf(11)

    
    for topic in topic_lst:
        print("Building model "+ str(doc_num) + " with " + str(topic) + " topics...")
        model.build_LDA(topic, passes=5)
        perplexities[idx].append(model.get_perplexity())

    
    

plot_perplexities(topic_lst, perplexities, doc_nums)

        
print(perplexities)