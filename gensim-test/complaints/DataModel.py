import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import gensim
import json

class DataModel:
    def __init__(self, path):
        self.path = path
        self.corpus = None
    
    def clean_data(self, doc_num=0):
        if self.corpus != None:
            return self.corpus

        # loading the data from the path 
        with open(self.path) as the_file:
            file_contents = the_file.read()
        parsed_json = json.loads(file_contents)

        # cleaning documents of stop words
        stop_words = stopwords.words('english')
        clean_docs = []
        doc_count = 0

        for entry in parsed_json:

            text = entry["_source"]["complaint_what_happened"]
            if text == "":
                continue

            text = text.splitlines()

            done = gensim.utils.simple_preprocess(" ".join(text))

            without_stops = [word for word in done if word not in stop_words]
            # remove the X-parts where information has been anonymized
            without_x = [ word for word in without_stops if not word.startswith("xx")]
            clean_docs.append(without_x)
            doc_count += 1
            
            # surveill progress
            if doc_count % 100:
                print("Documents cleaned: " + str(doc_count))
            # aquire only as many documents as you want
            if doc_num > 0 and doc_count == doc_num:
                break
        
        self.corpus = clean_docs
        return self.corpus
    

    def drop_words_on_tfidf(self, my_dict, tfidfs, thresh):
        drop_words = [my_dict[x] for x in my_dict.cfs.keys() if tfidfs[x] <= thresh]
        clean_docs = [[word for word in doc if doc not in drop_words] for doc in self.corpus]
        self.corpus = clean_docs
        return self.corpus

    def get_avg_doc_len(self):
        return sum([len(doc) for doc in self.corpus]) / len(self.corpus)

    def persist(self):
        pass


    


if __name__ == "__main__":
    data = DataModel("gensim-test/complaints/raw_data.json")
    data.clean_data()
    
