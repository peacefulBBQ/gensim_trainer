import matplotlib.pyplot as plt
import re
import os
import numpy as np
import json

class Plotter():
    def __init__(self, project_dir) -> None:
        os.chdir("generalized/classes")
        self.dir = project_dir
        self.pattern = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
        self.model_names = os.listdir(project_dir + "/models")




    def _demarker(self, name, pos):
        return  "----" + name + "----" + pos + "----"


    def _plot_convergence(self, model_name, likelihoods):
        ''' plot convergence for a single model and save it'''
        iter = list(range(0,len(likelihoods)*200,200)) # number (200) must fit with the eval_every value
        plt.plot(iter,likelihoods,c="black")
        plt.ylabel("log likelihood")
        plt.xlabel("iteration")
        plt.title("Topic Model Convergence")
        plt.ylim(bottom=0)
        plt.grid()
        plt.savefig(self.dir + "/models/" + model_name + "/plots" + "/convergence.pdf")
        plt.close()


    def plot_convergences(self):
        likelihoods = []
        reading = False
        
        with open(self.dir + "/gensim.log") as file:
            
            for text in file:
                match = self.pattern.findall(text)

                if re.search(self._demarker(".+", "START") + "$", text):
                    reading = True

                if re.search(self._demarker(".+", "STOP") + "$", text):
                    model_name = text.split("----")[-3] # this is the model name
                    self._plot_convergence(model_name, likelihoods)
                    reading = False
                    likelihoods = []


                if reading and len(match) != 0:
                    likelihoods.append(-float(match[0][0]))
                    # perpexities.append(mathch[0][1])
                    matches = [self.pattern.findall(l) for l in open(self.dir + '/gensim.log')]
                    matches = [m for m in matches if len(m) > 0]


    def plot_perplexities(self, topic_lst, perplexities, doc_nums):
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

    
    def plot_features(self, title, x_feat, y_feat, labelling):
        X = []
        Y = []
        labels = []

        for name in self.model_names:
            file = open(self.dir + "/models/" + name + "/measurements.txt")
            measures = json.loads(file.read())
            x = measures[x_feat]
            y = measures[y_feat]
            print(x, y, name)
            plt.plot(x, y, "bo")

            if labelling:
                plt.text(x * (1 + 0.001), y * (1 + 0.001) , name, fontsize=12)
                pass
        
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.title(title)
        plt.savefig(self.dir + "/figures/" + title)
        plt.close()
            


if __name__ == "__main__":
    my_plotter = Plotter("Plot Tester")
    # my_plotter.plot_convergences()

    my_plotter.plot_features("Check it", "coherence", "perplexity", True)


