from NMVM import NMVM
import time
import argparse
import os
import logging
import random
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='NMVM')
    parser.add_argument('--dataname', type=str, default='Tweet', help='dataset name')
    parser.add_argument('--_lambda', type=float, default=0.6, help='CMM and GMM weight')
    parser.add_argument('--K', type=int, default=0, help='initial number of clusters')
    parser.add_argument('--alpha', type=float, default=0.1, help='CMM prior parameter 1')
    parser.add_argument('--beta', type=float, default=0.05, help='CMM prior parameter 2')
    parser.add_argument('--sigma', type=float, default=0.0004, help='GMM prior parameter 1')
    parser.add_argument('--kappa', type=float, default=0.5, help='GMM prior parameter 2')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--iterNum', type=int, default=15, help='single trial model iterations')
    parser.add_argument('--sampleNum', type=int, default=5, help='repeat times')
    parser.add_argument('--wordsInTopicNum', type=int, default=20, help='topN clusters representative words')
    parser.add_argument('--dataDir', type=str, default='./data-GPT3s/', help='data directory')
    parser.add_argument('--outputPath', type=str, default='./results/', help='output path')
    return parser

def runNMVM(args):
    NMVM_model = NMVM(args.K, args.alpha, args.beta, args.sigma, args.kappa, args._lambda, 
                      args.iterNum, args.sampleNum, args.dataname, args.wordsInTopicNum, args.dataDir, args.seed)

    FinalMetrics = {"ACC":[], "NMI":[]}
    for sampleNo in range(1, args.sampleNum + 1):
        random.seed(args.seed + sampleNo)
        NMVM_model.getDocuments()
        print("SampleNo:" + str(sampleNo))
        ACC, NMI = NMVM_model.runNMVM(sampleNo, args.outputPath)
        FinalMetrics["ACC"].append(ACC)
        FinalMetrics["NMI"].append(NMI)

    return FinalMetrics


def main(args):
    outf = open("time_NMVM", "a")
    os.makedirs(args.outputPath, exist_ok=True)
    logging.basicConfig(filename=f'{args.outputPath}/log.txt', level=logging.INFO, 
                        format='%(asctime)s - %(message)s', encoding='utf-8')
    
    time1 = time.time()
    metrics = runNMVM(args)
    time2 = time.time()
    
    acc_mean = np.mean(metrics["ACC"])
    acc_std = np.std(metrics["ACC"])
    nmi_mean = np.mean(metrics["NMI"])
    nmi_std = np.std(metrics["NMI"])
    print(f"   Lambda = {args._lambda:.1f} completed!")
    print(f"   ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"   NMI: {nmi_mean:.4f} ± {nmi_std:.4f}")
    print(f"   Time: {(time2-time1)/5:.2f}s")
    
    outf.write("{}K{}alpha{:.4f}beta{:.4f}sigma{:.5f}kappa{:.3}_lambda{:.2f}iterNum{}SampleNum{}\ttime:{}".format(
        args.dataname, args.K, args.alpha, args.beta, args.sigma, args.kappa, args._lambda, args.iterNum, args.sampleNum, (time2-time1)
    ))


if __name__ == '__main__':

    parser = parse_arguments()
    args = parser.parse_args()

    main(args)
