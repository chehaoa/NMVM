from DocumentSet import DocumentSet
from Model import Model


class NMVM:

    def __init__(self, K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir):
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

        self.wordList = []
        self.wordToIdMap = {}

    def getDocuments(self):
        self.documentSet = DocumentSet(self.dataDir + self.dataset, self.wordToIdMap, self.wordList)
       
        self.V = self.wordToIdMap.__len__() 
        

    def runNMVM(self, sampleNo, outputPath):
        # ParametersStr = "K" + str(self.K) + "alpha" + str(round(self.alpha, 3)) + "beta" + str(round(self.beta, 3)) + \
        #                 "sigma" + str(self.sigma) + "kappa" + str(round(self.kappa, 3)) + "_lambda" + str(self._lambda) + \
        #                 "iterNum" + str(self.iterNum) + "SampleNum" + str(self.sampleNum)


        ParametersStr = "K{}alpha{:.4f}beta{:.4f}sigma{:.5f}kappa{:.3}_lambda{:.2f}iterNum{}SampleNum{}".format(
                        self.K,
                        self.alpha,
                        self.beta,
                        self.sigma,
                        self.kappa,
                        self._lambda,
                        self.iterNum,
                        self.sampleNum,
                    )
        model = Model(self.K, self.V, self.iterNum, self.alpha, self.beta, self.sigma, self.kappa, self._lambda,
                      self.dataset, ParametersStr, sampleNo, self.wordsInTopicNum)
        model.run_ICMM_withGMM(self.documentSet, outputPath, self.wordList)
