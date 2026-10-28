from NMVM import NMVM
import time
import argparse
import os
import logging
import random
import numpy as np


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def parse_arguments():
    parser = argparse.ArgumentParser(description='NMVM')
    parser.add_argument('--dataname', type=str, default='Tweet', help='dataset name')
    parser.add_argument('--lambda', '--_lambda', dest='_lambda', type=float, default=0.6,
                        help='bag-of-words view weight (embedding view uses 1 - lambda)')
    parser.add_argument('--K', type=int, default=0, help='initial number of clusters')
    parser.add_argument('--alpha', type=float, default=0.1, help='CMM prior parameter 1')
    parser.add_argument('--beta', type=float, default=0.05, help='CMM prior parameter 2')
    parser.add_argument('--sigma', type=float, default=0.0004, help='GMM prior parameter 1')
    parser.add_argument('--kappa', type=float, default=0.5, help='GMM prior parameter 2')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--iterNum', type=positive_int, default=15,
                        help='fixed number of full Gibbs sweeps after initialization (no early stopping)')
    parser.add_argument('--sampleNum', type=positive_int, default=5, help='repeat times')
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
    print(f"   Time per run: {(time2-time1)/args.sampleNum:.2f}s")
    
    with open("time_NMVM", "a", encoding="utf-8") as outf:
        outf.write("{}K{}alpha{:.4f}beta{:.4f}sigma{:.5f}kappa{:.3}_lambda{:.2f}iterNum{}SampleNum{}\ttime:{}\n".format(
            args.dataname, args.K, args.alpha, args.beta, args.sigma, args.kappa, args._lambda, args.iterNum, args.sampleNum, (time2-time1)
        ))


if __name__ == '__main__':

    parser = parse_arguments()
    args = parser.parse_args()

    main(args)
