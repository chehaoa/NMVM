from DocumentSet import DocumentSet
from Model import Model
import os

class NMVM:

    def __init__(self, K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir, seed):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.kappa = kappa
        self._lambda = _lambda
        self.iterNum = iterNum
        self.sampleNum = sampleNum
        self.dataset = dataset
        self.wordsInTopicNum = wordsInTopicNum
        self.dataDir = dataDir

        self.seed = seed
        self.wordList = []
        self.wordToIdMap = {}

    def getDocuments(self):
        data_path = self.dataDir + self.dataset
        if not os.path.exists(data_path):
            data_path += ".jsonl"
        self.documentSet = DocumentSet(data_path, self.wordToIdMap, self.wordList)

        self.V = self.wordToIdMap.__len__() 


    def runNMVM(self, sampleNo, outputPath):
        ParametersStr = "K{}alpha{:.4f}beta{:.4f}sigma{:.5f}kappa{:.3}_lambda{:.2f}iterNum{}SampleNum{}seed{}".format(
                        self.K,
                        self.alpha,
                        self.beta,
                        self.sigma,
                        self.kappa,
                        self._lambda,
                        self.iterNum,
                        self.sampleNum,
                        self.seed
                    )
        model = Model(self.K, self.V, self.iterNum, self.alpha, self.beta, self.sigma, self.kappa, self._lambda,
                      self.dataset, ParametersStr, sampleNo, self.wordsInTopicNum, self.seed)
        ACC, NMI = model.run_NMVM(self.documentSet, outputPath, self.wordList)
        return ACC, NMI
