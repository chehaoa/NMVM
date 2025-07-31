# -*- coding: utf-8 -*-
import codecs
import json
from sklearn import metrics
import numpy as np
import math


def cluster_acc(Y, Y_pred):
    from scipy.optimize import linear_sum_assignment as linear_assignment
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[ind[0][i], ind[1][i]] for i in range(D)]) * 1.0 / Y_pred.size


def cluster_purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def MeanAndVar(dataList):
    mean = sum(dataList)*1.0 / len(dataList)
    varience = math.sqrt(sum((mean - value) ** 2 for value in dataList)*1.0 / len(dataList))
    return (mean, varience)

class ClusterEvaluation():
    def __init__(self, resultFilePath):
        self.tweetsCleaned = []
        self.resultFilePath = resultFilePath

        self.ACCList = []
        self.purityList = []
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []
        self.SCList = []

        self.ACCTopicKList = []
        self.purityTopicKList = []
        self.AMITopicKList = []
        self.NMITopicKList = []
        self.MITopicKList = []
        self.ARITopicKList = []
        self.homogeneityTopicKList = []
        self.completenessTopicKList = []
        self.VTopicKList = []
        self.SCTopicKList = []
        self.docNum = 0
        self.KRealNum = -1

        self.labelsPred = {}
        self.labelsTrue = {}

    # Get labelsPred.
    def getPredLabels(self, inFile):
        self.labelsPred = {}
        with codecs.open(inFile, 'r') as fin:
            for line in fin:
                try:
                    documentID = line.strip().split()[0]
                    clusterNo = line.strip().split()[1]
                    self.labelsPred[int(documentID)] = int(clusterNo)
                except:
                    print(line)

    # Get labelsTrue and docs.
    def getTrueLabels(self, dataset):
        self.labelsTrue = {}
        self.docs = []
        with codecs.open(dataset, 'r') as fin:
            for docJson in fin:
                try:
                    docObj = json.loads(docJson)
                    self.labelsTrue[int(docObj['tweetId'])] = int(docObj['clusterNo'])
                    self.docs.append([int(docObj['tweetId']), docObj['textCleaned']])
                except:
                    print(docJson)
        self.KRealNum = len(set(self.labelsTrue.values()))

    def getKPredNum(self, inFile):
        labelsPred = []
        with codecs.open(inFile, 'r') as fin:
            for lineJson in fin:
                resultObj = json.loads(lineJson)
                labelsPred.append(resultObj['predictedCluster'])
        KPredNum = np.unique(labelsPred).shape[0]
        return KPredNum

    # Get scaled cluster number and scaled document number. Kthreshold is minimum number of a scaled cluster.
    def getPredNumThreshold(self, inFile, Kthreshold):
        KPredNum = 0
        docRemainNum = 0
        docTotalNum = 0
        with codecs.open(inFile, 'r', 'utf-8') as fin:
            clusterSizeStr = fin.readline().strip().strip(',')
        clusterSizeList = clusterSizeStr.split(',\t')
        for clusterSize in clusterSizeList:
            try:
                clusterSizeCouple = clusterSize.split(':')
                docTotalNum += int(clusterSizeCouple[1])
                if int(clusterSizeCouple[1]) > Kthreshold:
                    KPredNum += 1
                    docRemainNum += int(clusterSizeCouple[1])
            except:
                pass
        return (KPredNum, docRemainNum, docTotalNum)

    # Get evaluation of each sample
    def evaluatePerSample(self, sampleNo):
        labelsTrue = []
        labelsPred = []
        for d in self.docs:
            documentID = d[0]
            if documentID in self.labelsPred:
                labelsTrue.append(self.labelsTrue[documentID])
                labelsPred.append(self.labelsPred[documentID])
        ACC = cluster_acc(labelsTrue, labelsPred)
        purity = cluster_purity(labelsTrue, labelsPred)
        AMI = metrics.adjusted_mutual_info_score(labelsTrue, labelsPred)
        NMI = metrics.normalized_mutual_info_score(labelsTrue, labelsPred)
        MI = metrics.mutual_info_score(labelsTrue, labelsPred)
        ARI = metrics.adjusted_rand_score(labelsTrue, labelsPred)
        homogeneity = metrics.homogeneity_score(labelsTrue, labelsPred)
        completeness = metrics.completeness_score(labelsTrue, labelsPred)
        V = metrics.v_measure_score(labelsTrue, labelsPred)

        self.ACCList.append(ACC)
        self.purityList.append(purity)
        self.AMIList.append(AMI)
        self.NMIList.append(NMI)
        self.MIList.append(MI)
        self.ARIList.append(ARI)
        self.homogeneityList.append(homogeneity)
        self.completenessList.append(completeness)
        self.VList.append(V)

    # Get mean and var of all evaluation
    def evaluateAllSamples(self):
        self.ACCTopicKList.append(MeanAndVar(self.ACCList))
        self.purityTopicKList.append(MeanAndVar(self.purityList))
        self.ARITopicKList.append(MeanAndVar(self.ARIList))
        self.MITopicKList.append(MeanAndVar(self.MIList))
        self.AMITopicKList.append(MeanAndVar(self.AMIList))
        self.NMITopicKList.append(MeanAndVar(self.NMIList))
        self.homogeneityTopicKList.append(MeanAndVar(self.homogeneityList))
        self.completenessTopicKList.append(MeanAndVar(self.completenessList))
        self.VTopicKList.append(MeanAndVar(self.VList))

        self.ACCList = []
        self.purityList = []
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []

    def drawEvaluationResult(self, titleStr):
        ACCMeanList = [item[0] for item in self.ACCTopicKList]
        purityMeanList = [item[0] for item in self.purityTopicKList]
        ARIMeanList = [item[0] for item in self.ARITopicKList]
        AMIMeanList = [item[0] for item in self.AMITopicKList]
        NMIMeanList = [item[0] for item in self.NMITopicKList]
        homogeneityMeanList = [item[0] for item in self.homogeneityTopicKList]
        completenessMeanList = [item[0] for item in self.completenessTopicKList]
        VMeanList = [item[0] for item in self.VTopicKList]
        with open(self.resultFilePath, 'w') as fout:
            fout.write('Parameter: ' + repr(titleStr) + '\n')
            fout.write('ACCMean: ' + repr(ACCMeanList) + '\n')
            fout.write('purityMean: ' + repr(purityMeanList) + '\n')
            fout.write('ARIMean: ' + repr(ARIMeanList) + '\n')
            fout.write('AMIMean: ' + repr(AMIMeanList) + '\n')
            fout.write('NMIMean: ' + repr(NMIMeanList) + '\n')
            fout.write('homogeneityMean: ' + repr(homogeneityMeanList) + '\n')
            fout.write('completenessMean: ' + repr(completenessMeanList) + '\n')
            fout.write('VMean: ' + repr(VMeanList) + '\n')
            fout.write('\n')

        ACCVarianceList = [item[1] for item in self.ACCTopicKList]
        purityVarianceList = [item[1] for item in self.purityTopicKList]
        ARIVarianceList = [item[1] for item in self.ARITopicKList]
        AMIVarianceList = [item[1] for item in self.AMITopicKList]
        NMIVarianceList = [item[1] for item in self.NMITopicKList]
        homogeneityVarianceList = [item[1] for item in self.homogeneityTopicKList]
        completenessVarianceList = [item[1] for item in self.completenessTopicKList]
        VVarianceList = [item[1] for item in self.VTopicKList]
        with open(self.resultFilePath, 'a') as fout:
            fout.write('ACCVariance: ' + repr(ACCVarianceList) + '\n')
            fout.write('purityVariance: ' + repr(purityVarianceList) + '\n')
            fout.write('ARIVariance: ' + repr(ARIVarianceList) + '\n')
            fout.write('AMIVariance: ' + repr(AMIVarianceList) + '\n')
            fout.write('NMIVariance: ' + repr(NMIVarianceList) + '\n')
            fout.write('homogeneityVariance: ' + repr(homogeneityVarianceList) + '\n')
            fout.write('completenessVariance: ' + repr(completenessVarianceList) + '\n')
            fout.write('VVariance: ' + repr(VVarianceList) + '\n')
            fout.write('\n')

    def drawPredK(self, KRealNumMeanList, KPredNumMeanList, KRealNumVarianceList, KPredNumVarianceList):
        with open(self.resultFilePath, 'a') as fout:
            fout.write('KRealNumMean: ' + repr(KRealNumMeanList) + '\n')
            fout.write('KPredNumMean: ' + repr(KPredNumMeanList) + '\n')
            fout.write('\n')
            fout.write('KRealNumVariance: ' + repr(KRealNumVarianceList) + '\n')
            fout.write('KPredNumVariance: ' + repr(KPredNumVarianceList) + '\n')
            # fout.write('\n' + '-' * 50 + '\n\n')


def ICMM_withGMMEvaluation():
    dirName = '%sK%dalpha%sbeta%ssigma%skappa%s_lambda%siterNum%dSampleNum%d/' % \
              (dataset, K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum)
    resultFileName = 'NMVCDataset%sK%dalpha%sbeta%ssigma%skappa%s_lambda%siterNum%dSampleNum%dNoiseKThreshold%d.yaml' % (
        dataset, K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, KThreshold)
    resultFilePath = inPath + dirName + resultFileName
    ICMM_withGMMEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    KRealNumMeanList = []
    KRealNumVarianceList = []
    noiseNumMeanList = []
    noiseNumVarianceList = []
    KPredNumList = []
    KRealNumList = []
    noiseNumList = []

    for sampleNo in range(1, sampleNum + 1):
        inDir = inPath + dirName
        fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
        inFile = inDir + fileName
        ICMM_withGMMEvaluation.getPredLabels(inFile)
        sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
        (KPredNum, docRemainNum, docTotalNum) = ICMM_withGMMEvaluation.getPredNumThreshold(sizeFile, KThreshold)
        ICMM_withGMMEvaluation.getTrueLabels(datasetPath)

        ICMM_withGMMEvaluation.evaluatePerSample(sampleNo)
        KPredNumList.append(KPredNum)
        KRealNumList.append(ICMM_withGMMEvaluation.KRealNum)
        noiseNumList.append(docTotalNum - docRemainNum)

    ICMM_withGMMEvaluation.evaluateAllSamples()
    KPredNumMeanList.append(np.mean(KPredNumList))      # mean
    KRealNumMeanList.append(np.mean(KRealNumList))
    noiseNumMeanList.append(np.mean(noiseNumList))
    KPredNumVarianceList.append(np.std(KPredNumList))   # standard deviation
    KRealNumVarianceList.append(np.std(KRealNumList))
    noiseNumVarianceList.append(np.std(noiseNumList))

    titleStr = 'NMVC %s K%d iterNum%d SampleNum%d alpha%s beta%s sigma%s kappa%s _lambda%s' % \
               (dataset, K, iterNum, sampleNum, alpha, beta, sigma, kappa, _lambda)
    ICMM_withGMMEvaluation.drawEvaluationResult(titleStr)
    ICMM_withGMMEvaluation.drawPredK(KRealNumMeanList, KPredNumMeanList, KRealNumVarianceList, KPredNumVarianceList)


if __name__ == '__main__':
    KThreshold = 0
    sampleNum = 2
    iterNum = 15
    
    K = 0
    alpha = '{:.4f}'.format(0.1)
    
    beta = "{:.4f}".format(0.05)
  
    sigma = "{:.5f}".format(0.0004)
    
    kappa = "{:.1f}".format(0.5)

    _lambda = "{:.2f}".format(0.6)
    

   
    dataset = "Tweet-SIMCSE"
    datasetPath = '../data/' + dataset
    inPath = './result-Tweet/'

    ICMM_withGMMEvaluation()

   