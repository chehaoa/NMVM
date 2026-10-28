import json
from Document import Document
import numpy as np
import pandas as pd
import random


class DocumentSet:

    def __init__(self, dataDir, wordToIdMap, wordList):
        self.D = 0
        self.documents = []
        self.mean_embedding = None

        self.labels = []

        with open(dataDir, encoding="utf-8") as input:
            all_lines = input.readlines()

        random.shuffle(all_lines)

        for line in all_lines:
            if line.strip():
                self.D += 1
                obj = json.loads(line)
                text = obj['textCleaned']
                embedding = np.array(obj['embedding'])
                if self.mean_embedding is None:
                    self.mean_embedding = np.zeros_like(embedding, dtype=np.float64)
                self.mean_embedding += embedding
                document = Document(text, wordToIdMap, wordList, self.D, embedding)
                self.documents.append(document)
                if 'clusterNo' in obj:
                    self.labels.append(int(obj['clusterNo']) - 1)
                else:
                    self.labels.append(int(obj['cluster_id']) - 1)
        
        self.labels = pd.Series(self.labels)
        # print("labels min and max: ", self.labels.min(), self.labels.max())

        self.mean_embedding /= self.D
        print("number of documents is ", self.D)
