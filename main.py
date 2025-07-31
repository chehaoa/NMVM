"""
ICMM_withGMM
"""

from NMVM import NMVM
import time


K = 0                  
sampleNum = 5         
iterNum = 15          
wordsInTopicNum = 20  

alpha = 0.1          
beta = 0.05          

sigma = 0.0004    
kappa = 0.5                         

_lambda = 0.6  


dataset = "Tweet-SIMCSE"

dataDir = "./data/"
outputPath = "./result-Tweet/"


def runNMVM(K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir):
    nmvm = NMVM(K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
   
    
    nmvm.getDocuments()
    for sampleNo in range(1, sampleNum + 1):
        print("SampleNo:" + str(sampleNo))
        nmvm.runICMM_withGMM(sampleNo, outputPath)

if __name__ == '__main__':
    outf = open("time_ICMM_withGMM", "a")
    time1 = time.time()
    runNMVM(K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
    time2 = time.time()
    # outf.write(str(dataset) + "K" + str(K) + "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) +
    #            "sigma" + str(round(sigma, 4)) + "kappa" + str(round(kappa, 3)) + "_lambda" + str(_lambda) +
    #            "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) +
    #            "\ttime:" + str(time2 - time1) + "\n")
    
    outf.write("{}K{}alpha{:.4f}beta{:.4f}sigma{:.5f}kappa{:.3}_lambda{:.2f}iterNum{}SampleNum{}\ttime:{}".format(
        dataset, 
        K,
        alpha,
        beta,
        sigma,
        kappa,
        _lambda,
        iterNum,
        sampleNum,
        (time2-time1)
    ))


    



