# -*- coding: utf-8 -*-

from __future__ import division
from numpy import mean, std, array, argpartition,count_nonzero, empty, argsort
import matplotlib.pyplot as plt  
from math import sqrt
from itertools import starmap, islice, cycle
import os, arff
from sklearn.metrics import roc_curve, auc
from scipy.io import loadmat
from sklearn.metrics import auc, precision_score, recall_score, f1_score

def loadData(fileName,data_type, str): 
    point_set = [] 
    for line in open(fileName, 'r'): 
        point = [data_type(data) for data in line.split(str)]
        point_set.append(point)
    return array(point_set) 

def dist(point1,point2):
    sum_dis = 0.0
    dimension = len(point1)
    for index in range(dimension)  :
        sum_dis += (point2[index] - point1[index])**2
    return sqrt(sum_dis)

# case2: -2: id  -1: outlier 
def load_arff2(fileName):
    with open(fileName) as fh:
        dataset = array(arff.load(fh)['data'])
        point_set = dataset[:,:-2].astype(float)
        labels = dataset[:,-1]
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no' :
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(int), outlier_num

def scaled_mst(point_set):
    result_set = []; nodes_finished = []; nodes_unfinished = []
    ps_size = len(point_set)
    ratio_arr = [1] * ps_size
    dist_arr = [0] * ps_size
    edge_arr = [-1]* ps_size
    temp_dist1 = 1.0e14; position = -1; s= 0
    nodes_finished.append(s)
    for index in range(len(point_set)):
        if index == s :
            continue
        t = dist(point_set[s], point_set[index])
        if t == 0 :
            result_set.append([s, index, 0])
            nodes_finished.append(index)
        else:
            dist_arr[index] = t
            edge_arr[index] = s
            ratio_arr[index] = t 
            if t < temp_dist1 :
                temp_dist1 = t
                position = index
    nodes_finished.append(position)
    result_set.append([s,position,temp_dist1])
    for index in range(len(point_set)):
        ratio_arr[index] = dist_arr[index] / temp_dist1
        if index not in nodes_finished :
            nodes_unfinished.append(index)
    q_index = 0
    while len(nodes_finished) < ps_size :
        min_ratio = 1.0e14
        for point_i in nodes_unfinished :
            new_node = nodes_finished[-1]
            d = dist(point_set[new_node], point_set[point_i])
            if d == 0 :
                result_set.append([new_node, point_i, 0])
                nodes_finished.append(point_i)
                nodes_unfinished.remove(point_i)
                continue
            r = d / temp_dist1
            if r < ratio_arr[point_i] : 
                dist_arr[point_i] = d
                ratio_arr[point_i] = r
                edge_arr[point_i] = new_node
            if ratio_arr[point_i] < min_ratio  :
                min_ratio = ratio_arr[point_i]
                q_index = point_i
        temp_dist1 = dist_arr[q_index]
        nodes_finished.append(q_index)
        nodes_unfinished.remove(q_index)
        result_set.append([edge_arr[q_index], q_index, ratio_arr[q_index]])
    return result_set, edge_arr, ratio_arr

def dfs(T, x, adjencent):
    for p in adjencent[x] :
        if p not in T :
            T.append(p)
            dfs(T, p, adjencent)

def cut_edge(edge, adjencent):
    Tu = []; Tv = []
    adjencent[edge[0]].remove(edge[1])
    adjencent[edge[1]].remove(edge[0])
    Tu.append(edge[0])
    Tv.append(edge[1])
    dfs(Tu, edge[0], adjencent)
    dfs(Tv, edge[1], adjencent)
    return Tu, Tv

def get_mean_std(edge_set):
    sum_dist = 0; std = 0
    n = len(edge_set )
    for edge in edge_set:
        sum_dist += edge[2]
    mean = sum_dist / n
    for edge in edge_set:
        std += abs(edge[2] - mean) ** 2
    std = sqrt(std)
    return mean + std

def cut_tree(tree, adjencent, clusters, largest_point):
    tu, tv = cut_edge(tree[0], adjencent)
    tree.remove(tree[0])
    left_tree = []; right_tree = []
    if len(tu) > largest_point :
        for edge in tree:
            if edge[0] in tu or edge[1] in tu :
                left_tree.append(edge)
        cut_tree(left_tree, adjencent, clusters, largest_point)
    else:
        clusters.append(tu)
    if len(tv) > largest_point :
        for edge in tree:
            if edge[0] in tv or edge[1] in tv :
                right_tree.append(edge)
        cut_tree(right_tree, adjencent, clusters, largest_point)
    else:
        clusters.append(tv)

def CMOD(point_set):
    dimension = len(point_set[0])
    result_set, edge_arr, dist_arr = scaled_mst(point_set)
    sorted_edge = sorted(result_set, key = lambda x:x[2], reverse=True)
    data_size = len(point_set)
    least_point = sqrt(data_size / dimension)
    largest_point = data_size - least_point
    ratio_threshold = get_mean_std(sorted_edge)
    labels = [0]*data_size
    adjencent = [[] for i in range(data_size)]  
    for edge in sorted_edge:
        adjencent[edge[0]].append(edge[1])
        adjencent[edge[1]].append(edge[0])
    clusters = []
    cut_tree(sorted_edge, adjencent, clusters, largest_point)
    cls_num = len(clusters); centroids = [0]*cls_num
    scores = [0]* len(point_set)
    for i, cl in enumerate(clusters):
        if len(cl) < least_point :
            for p in cl:
                scores[p] = 1e14
            continue
        temp_centroid = get_centroid(cl, point_set)
        for p in cl:
            scores[p] = dist(point_set[p], temp_centroid)
    return scores

def scores2outliers(scores, outlier_num):
    # scores_arr = array(scores)
    # outliers = argpartition(scores_arr, outlier_num)
    # print(outliers[-outlier_num:])
    sorted_scores = sorted(scores, reverse=True)
    # print(sorted_scores)
    outliers = []
    # outliers = scores_arr.argmin(numberofvalues=outlier_num) 
    for i in range(outlier_num):
        idx = scores.index(sorted_scores[i])
        scores[idx] = 0
        outliers.append(idx)
        # scores.remove(scores[idx])
    return outliers

def get_centroid(clusters, point_set):
    sum_dist = [0] * len(clusters)
    for i, p in enumerate(clusters):
        for j, q in enumerate(clusters):
            sum_dist[i] += dist(point_set[p], point_set[q])
    return point_set[clusters[sum_dist.index(min(sum_dist))]]

# case3: -2: outlier -1: id 
def load_arff3(fileName):
    with open(fileName) as fh:
        dataset = array(arff.load(fh)['data'])
        point_set = dataset[:,:-2].astype(float)
        labels = dataset[:,-2]
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no' :
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(int), outlier_num

# case4: 0 : id; -1: outlier
def load_arff4(fileName):
    with open(fileName) as fh:
        dataset = array(arff.load(fh)['data'])
        point_set = dataset[:,1:-1].astype(float)
        labels = dataset[:,-1]
        outlier_num = 0
        for i, l in enumerate(labels):
            if l == 'no' :
                labels[i] = 0
            else:
                labels[i] = 1
                outlier_num += 1
    return point_set, labels.astype(int), outlier_num

def get_centroid1_ps(clusters, point_set):
    centroids = [0] * len(point_set[0])
    for p in clusters:
        for i, ele in enumerate(point_set[p]):
            centroids[i] += ele
    center = [ce/ len(clusters) for ce in centroids]
    return center

def get_centroid1(clusters):
    centroids = [0] * len(clusters[0])
    for p in clusters:
        for i, ele in enumerate(p):
            centroids[i] += ele
    center = [ce/ len(clusters) for ce in centroids]
    return center

def get_centroid2(clusters):
    sum_dist = [0] * len(clusters)
    for i, p in enumerate(clusters):
        for j, q in enumerate(clusters):
            sum_dist[i] += dist(p, q)
    return clusters[sum_dist.index(min(sum_dist))].tolist()

def prim_mst(point_set):
    result_set = []
    nodes_finished = []; nodes_unfinished = []
    nodes_finished.append(0)
    dist_arr = [0] * len(point_set)
    edge_arr = [-1]* len(point_set)
    temp_dist1 = 1.0e14; position = -1
    for index in range(len(point_set)):
        if index == 0 :
            continue
        t = dist(point_set[0], point_set[index])
        dist_arr[index] = t
        edge_arr[index] = 0
        if t < temp_dist1 :
            temp_dist1 = t
            position = index
    nodes_finished.append(position)
    result_set.append([0, position, temp_dist1])
    for index in range(len(point_set)):
        if index != 0 and index != position :
            nodes_unfinished.append(index)
    q_index = 0
    while len(nodes_unfinished) > 0 :
        temp_dist2 = 1.0e14
        new_node = nodes_finished[-1]
        for point_i in nodes_unfinished :
            d = dist(point_set[new_node], point_set[point_i])
            if d < dist_arr[point_i] : #and r != 0 :
                dist_arr[point_i] = d
                edge_arr[point_i] = new_node
            if dist_arr[point_i] < temp_dist2  :
                temp_dist2 = dist_arr[point_i]
                q_index = point_i
        nodes_finished.append(q_index)
        nodes_unfinished.remove(q_index)
        result_set.append([edge_arr[q_index], q_index,dist_arr[q_index]])
    return result_set, edge_arr, dist_arr

def cent_score(center, point_set):
    scores = empty(len(point_set)) 
    for i,p in enumerate(point_set):
        print("p,center:",p,center)
        scores[i] = dist(p, center)
    print(center)
    print(scores)
    return argsort(scores)

if __name__ == "__main__" :
    # fileName = "../data/data27.dat"      # 2
    # point_set = loadData(fileName, float, ',')

    # outlier_num = 5
    # scores = CMOD(point_set)
    # outliers = scores2outliers(scores, outlier_num)
    # print(outliers)

    p = r'E:\\data\\arff2\\arff4'
    
    f1 = open("../result/method1_auc_%s.csv"%("WDBC"),'w')
    for root,dirs,files in os.walk(p): 
        for name in files:
            fileName = os.path.join(p,name)
            file_name, file_type = os.path.splitext(name)
            m = loadmat(fileName)
            point_set = m["X"]; labels = m["y"].ravel()
            # point_set, labels, outlier_num = load_arff2(fileName)
            # point_set, labels, outlier_num = load_cls1o(fileName)
            # point_set = np.array(point_set.tolist()).astype(np.float)
            # labels = np.array(labels).astype(int)
            # print(file_name, len(point_set), outlier_num, len(point_set[0]))
            scores = CMOD(point_set)
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr,tpr)
            print(file_name, "%0.4f"%(roc_auc))
            f1.write(file_name+','+'our'+','+ str("%0.4f,"%(roc_auc)) + '\n')
