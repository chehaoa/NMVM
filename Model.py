import random
import os
import copy
import math
import numpy as np
import sys
import logging
import pandas as pd
from tqdm import tqdm
from utils import cluster_accuracy

class Model:

    def __init__(self, K, V, iterNum, alpha, beta, sigma, kappa, _lambda, dataset, ParametersStr, sampleNo,
                 wordsInTopicNum, seed):
        self.K = K
        self.V = V
        self.iterNum = iterNum
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.kappa = kappa
        self.mu = None
        self._lambda = _lambda

        self.beta0 = float(V) * float(beta)

        self.smallDouble = 1e-150
        self.largeDouble = 1e150
        self.seed = seed

        self.large = 1e150
        self.small = 1e-150

    def run_NMVM(self, documentSet, outputPath, wordList):
        # The whole number of documents
        self.D_All = documentSet.D
        # hyper-parameter mu in gmm
        self.mu = documentSet.mean_embedding

        # Cluster assignments of each document               (documentID -> clusterID)
        self.z = [-1] * self.D_All
        # The embedding of cluster z                         (clusterID -> cluster embedding)
        self.e_z = [np.zeros_like(self.mu, dtype=np.float64) for _ in range(self.K)]
        # The number of documents in cluster z               (clusterID -> number of documents)
        self.m_z = [0] * self.K
        # The number of words in cluster z                   (clusterID -> number of words)
        self.n_z = [0] * self.K
        # The number of occurrences of word v in cluster z   (n_zv[clusterID][wordID] = number)
        self.n_zv = [[0] * self.V for _ in range(self.K)]
        # different from K, K is clusterID but K_current is clusterNum
        self.K_current = copy.deepcopy(self.K)
        # word list in initialization
        self.word_current = []

        self.sigma_k_list = []

        logging.info(f"Begin to initialize")
        
        self.intialize(documentSet)
        self.showTempResult(documentSet.labels)
        logging.info(f"Initialization successful! Begin to Gibbs sampling")
        self.gibbsSampling(documentSet)
        print("\tGibbs sampling successful! Start to saving results.")
        self.output(documentSet, outputPath, wordList)
        logging.info(f"Saving successful!")
        logging.info(f"End of the program")
        metrics = self.showTempResult(documentSet.labels)
        ACC = metrics['ACC']
        NMI = metrics['NMI']

        # self.check_sigma_k()
        print("\tSaving successful!")

        return ACC, NMI

    # Get beta0 for current V
    def getBeta0(self):
        return (float(len(list(set(self.word_current)))) * float(self.beta))
    
    def showTempResult(self, cluster_label):
        cluster_pred = pd.Series([self.z[d] for d in range(0, self.D_All)])
        metrics = cluster_accuracy(cluster_label, cluster_pred)
        # print(f"\t{stage} stage | Metrics:", metrics)
        logging.info(f"\tMetrics: {metrics} + K={self.K_current}")
        return metrics

    def intialize(self, documentSet):
        self.alpha0 = self.alpha * self.D_All
        print("\t" + str(self.D_All) + " documents will be analyze. alpha is" + " %.4f." % self.alpha +
              " beta is" + " %.4f." % self.beta + "\n\tsigma is" + " %.5f." % self.sigma +
              " kappa is" + " %.1f." % self.kappa + "\n\tlambda is" + " %.1f." % self._lambda +
              "\n\tK is" + " %.1f." % self.K +
              "\n\tInitialization.")

        for d in tqdm(range(0, self.D_All), desc="Initializing"):
            document = documentSet.documents[d]
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                self.word_current.append(wordNo)
            self.beta0 = self.getBeta0()
            if self.beta0 <= 0:
                print("Wrong V!")
                exit(-1)

            cluster = self.sampleCluster(-1, document, "Max")

            self.z[d] = cluster
            if cluster == len(self.m_z):
                self.m_z.append(0)
                self.n_zv.append([0] * self.V)
                self.n_z.append(0)
                self.e_z.append(np.zeros_like(document.embedding, dtype=np.float64))
            self.m_z[cluster] += 1
            self.e_z[cluster] += document.embedding
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                self.n_zv[cluster][wordNo] += wordFre
                self.n_z[cluster] += wordFre

    def gibbsSampling(self, documentSet):
        patience = 3
        no_improve_count = 0
        last_score = float('-inf')

        for i in range(self.iterNum):
            print("\titer is ", i + 1, end="\t")
            print("beta is" + " %f." % self.beta, end='\t')
            print("Kcurrent is" + " %f." % self.K_current, end='\n')
            if i != 0:
                metrics = self.showTempResult(documentSet.labels)
                current_score = metrics['NMI']
                if current_score > last_score:
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                last_score = current_score
                if no_improve_count >= patience:
                    break

            for d in tqdm(range(0, self.D_All), desc="Gibbs Sampling"):
                document = documentSet.documents[d]
                cluster = self.z[d]
                self.m_z[cluster] -= 1
                self.e_z[cluster] -= document.embedding
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)

                if i != self.iterNum - 1:
                    cluster = self.sampleCluster(i, document, "Max")
                else:
                    cluster = self.sampleCluster(i, document, "Max")

                self.z[d] = cluster
                if cluster == len(self.m_z):
                    self.m_z.append(0)
                    self.n_zv.append([0] * self.V)
                    self.n_z.append(0)
                    self.e_z.append(np.zeros_like(document.embedding, dtype=np.float64))
                self.m_z[cluster] += 1
                self.e_z[cluster] += document.embedding
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre


    def sumNormalization(self, x):
        x = np.array(x)
        norm_x = x / np.sum(x)
        return norm_x


    def sampleCluster(self, _iter, document, MODE):

        prob_1 = [float(0.0)] * (self.K + 1)
        overflowCount = [float(0.0)] * (self.K + 1)
        prob_2 = [float(0.0)] * (self.K + 1)
        e_index = [float(0.0)] * (self.K + 1)
        valueOfRule1 = [float(0.0)] * (self.K + 1)

        for k in range(self.K):
            if self.m_z[k] == 0:
                prob_1[k] = 0
                prob_2[k] = 0
                continue
            # dmm
            valueOfRule1[k] = self.m_z[k] / (self.D_All - 1 + self.alpha0)
            valueOfRule2 = 1.0
            i = 0
            for _, w in enumerate(range(document.wordNum)):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if valueOfRule2 < self.smallDouble:
                        overflowCount[k] -= 1
                        valueOfRule2 *= self.largeDouble
                    valueOfRule2 *= (self.n_zv[k][wordNo] + self.beta + j) / (self.n_z[k] + self.beta0 + i)
                    i += 1
            prob_1[k] = valueOfRule1[k] * valueOfRule2

            # gmm
            x = document.embedding
            sigma0 = self.sigma / self.kappa
            mu_k = (self.sigma * self.mu + sigma0 * self.e_z[k]) \
                   / (self.sigma + self.m_z[k] * sigma0)
            sigma_k = self.sigma + (self.sigma * sigma0) / (self.m_z[k] * sigma0 + self.sigma)
            x_sub_mu = np.array(x) - np.array(mu_k)
            distance_squared = np.dot(x_sub_mu, x_sub_mu)

            d = len(x)
            e_index[k] = -0.5 * (d * math.log(2 * math.pi * sigma_k) + distance_squared / sigma_k)

        '''==============================================================================='''
        '''the prob of new cluster'''
        # dmm
        valueOfRule1[self.K] = self.alpha0 / (self.D_All - 1 + self.alpha0)
        valueOfRule2 = 1.0
        i = 0
        for _, w in enumerate(range(document.wordNum)):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                if valueOfRule2 < self.smallDouble:
                    overflowCount[self.K] -= 1
                    valueOfRule2 *= self.largeDouble
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob_1[self.K] = valueOfRule1[self.K] * valueOfRule2
        
        # gmm
        x = document.embedding
        sigma0 = self.sigma / self.kappa
        mu_k = self.mu
        sigma_k = self.sigma + sigma0
        x_sub_mu = np.array(x) - np.array(mu_k)
        distance_squared = np.dot(x_sub_mu, x_sub_mu)

        d = len(x) 
        e_index[self.K] = -0.5 * (d * math.log(2 * math.pi * sigma_k) + distance_squared / sigma_k)


        max_overflow = -sys.maxsize
        for k in range(self.K + 1):
            if overflowCount[k] > max_overflow and prob_1[k] > 0.0:
                max_overflow = overflowCount[k]

        e_index = np.array(e_index)
        max_index = max(e_index[e_index!=0.0])
        for k in range(self.K + 1):
            if prob_1[k] > 0.0:
                prob_1[k] = prob_1[k] * math.pow(self.largeDouble, overflowCount[k] - max_overflow)
                prob_2[k] = valueOfRule1[k] * math.exp(e_index[k] - max_index)


        prob = [float(0.0)] * (self.K + 1)
        prob_1 = self.sumNormalization(prob_1)  # dmm
        prob_2 = self.sumNormalization(prob_2)  # gmm

        # weighted sum
        for i in range(self.K + 1):
            prob[i] = prob_1[i] * self._lambda + prob_2[i] * (1 - self._lambda)

        if MODE == "Random":
            kChoosed = 0
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
            return kChoosed

        elif MODE == "Max":
            kChoosed = 0
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
            return kChoosed

    # update K_current
    def check_sigma_k(self):
        mean = np.mean(self.sigma_k_list)
        var = np.var(self.sigma_k_list)
    def checkEmpty(self, cluster):

        if self.m_z[cluster] == 0:
            self.K_current -= 1

    def output(self, documentSet, outputPath, wordList):

        outputDir = outputPath + self.dataset + self.ParametersStr + "/"
        try:
            # create result/
            isExists = os.path.exists(outputPath)
            if not isExists:
                os.mkdir(outputPath)
                print("\tCreate directory:", outputPath)
            # create after result
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        try:
            self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        except:
            print("\tOutput Phi Words Wrong!")
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):  # φ
        self.phi_zv = [[0] * self.V for _ in range(self.K)]
        for cluster in range(self.K):
            for v in range(self.V):
                self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(
                    self.n_z[cluster] + self.beta0)

    def getTop(self, array, rankList, Cnt):

        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in range(len(array)):
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):

        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if self.m_z[k] == 0:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):

        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key=lambda tc: tc[1], reverse=True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):

        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(0, self.D_All):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[d]
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()
