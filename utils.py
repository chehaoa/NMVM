
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, homogeneity_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics.cluster import contingency_matrix as cm


def externalValidation(truthClusters, predictedClusters):
    def purity_score(y_true, y_pred):
        contingency_matrix_mine = cm(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix_mine, axis=0)) / np.sum(contingency_matrix_mine) 
    scores = {}
    scores['_adjusted_rand_index'] = adjusted_rand_score(truthClusters, predictedClusters)
    scores['_homogeneity_score'] = homogeneity_score(truthClusters, predictedClusters)
    scores['_purity_score'] = purity_score(truthClusters, predictedClusters)
    scores['_adjusted_mutual_info_score'] = adjusted_mutual_info_score(truthClusters, predictedClusters)
    scores['_fowlkes_mallows_score'] = fowlkes_mallows_score(truthClusters, predictedClusters)  
    return scores

def cluster_accuracy(y_true, y_pred):

    true_labels = list(set(y_true))
    pred_labels = list(set(y_pred))

    num_true_classes = len(true_labels)
    num_pred_classes = len(pred_labels)

    cost_matrix = np.zeros((num_true_classes, num_pred_classes), dtype=int)
    for i, true_label in enumerate(true_labels):
        true_indices = [index for index, label in enumerate(y_true) if label == true_label]
        for j, pred_label in enumerate(pred_labels):
            pred_indices = [index for index in true_indices if y_pred[index] == pred_label]
            cost_matrix[i][j] = -len(pred_indices)

    m = Munkres()
    indexes = m.compute(cost_matrix.tolist())
    total_correct = sum(-cost_matrix[i][j] for i, j in indexes)

    accuracy = total_correct / len(y_true)

    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    ari_score = adjusted_rand_score(y_true, y_pred)

    new_predict = np.zeros(len(y_pred))
    for i, (true_index, pred_index) in enumerate(indexes):
        new_predict[np.where(y_pred == pred_labels[pred_index])] = true_labels[true_index]
    
    f1 = f1_score(y_true, new_predict, average='macro')

    return {"ACC": accuracy, "NMI": nmi_score, "ARI": ari_score, "F1-score": f1}

def cluster_purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    purity_per_cluster = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        if len(idx) == 0:
            continue
        labels_tmp = labels_true[idx, :].reshape(-1)
        purity = np.bincount(labels_tmp).max() / len(labels_tmp)
        purity_per_cluster.append((c, purity))
        count.append(np.bincount(labels_tmp).max())
    
    total_purity = np.sum(count) / labels_true.shape[0]
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.countplot(x=labels_true.reshape(-1), palette="Set2")
    plt.title("True Labels Distribution")
    plt.xlabel("True Labels")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.countplot(x=labels_pred.reshape(-1), palette="Set2")
    plt.title("Predicted Labels Distribution")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("result_img/LD.png")

    cluster_labels, purity_values = zip(*purity_per_cluster)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=cluster_labels, y=purity_values, palette="Set1")
    plt.title("Purity per Cluster")
    plt.xlabel("Cluster Labels")
    plt.ylabel("Purity")
    plt.savefig("result_img/CL.png")

    return total_purity


