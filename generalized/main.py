from gensim.models import TfidfModel
from gensim.models import LdaModel

import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

sys.path.insert(0,"/Users/benschaefer/Flex/gensim_trainer/generalized/classes")

from DataModel import DataModel
from MyModel import MyModel
from Plotter import Plotter

def demarker(name, pos):
    return  "----" + name + "----" + pos + "----"

if __name__ == "__main__":



    # Initialization of a project
    project_name = "Plot Tester"
    

    # Create according folders
    print(os.getcwd())
    os.chdir("generalized")
    project_path = "projects/" + project_name



    if os.path.exists(project_path):
        print("Project exists already!")
    else:
        os.mkdir(project_path)
        os.mkdir(project_path + "/figures")
        os.mkdir(project_path + "/models")
        logging.basicConfig(filename= project_path + '/gensim.log',
                    format="%(asctime)s:%(levelname)s:%(message)s",
                    level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel("INFO")


    # Load the data
    data_model = DataModel("data/raw_data.json", 1000)


    # build the models
    os.chdir(project_path + "/models")
    models = []

    perplexities = []
    topic_lst = [10,20,30]
    doc_nums = [250,500,1000,2000]
    # This must be set for every project

    num_models = 3
    sizes = [1000, 1000, 1000]
    topics = [30, 30, 30]
    alphas =  ["auto", "symmetric", "asymmetric"]
    etas = ["auto", "symmetric", "symmetric"]
    params = [{"topics": topics[i], "alpha": alphas[i], "eta": etas[i], "passes": 5} for i in range(num_models)]
    
    for i in range(num_models):
        model = MyModel(data_model.draw_random(sizes[i]), alphas[i])
        model.save_params(params[i])
        print("Build model for " + str(alphas[i]))
        logger.info(demarker(alphas[i], "START"))
        model.build_LDA(**params[i])
        logger.info(demarker(alphas[i], "STOP"))
        model.save_measures()

        

    os.chdir("..")

    # plotter = Plotter()
    # plotter.plotting_convergence(".")

