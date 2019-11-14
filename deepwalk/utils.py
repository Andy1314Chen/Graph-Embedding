import random
import numpy as np
import networkx as nx

# 别名采样
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def read_graph(filename, is_weighted=False, is_directed=False):
    cateDict = dict()
    with open("data/cora.content", "r") as f:
        lines = f.readlines()    
        lines = list(map(lambda line: line.strip().split("\t"), lines))
    for line in lines:
        cateDict[int(line[0])] = line[-1]
    
    with open(filename, "r") as f:
        lines = f.readlines()
    lines = list(map(lambda line: [int(v) for v in line.strip().split("\t")], lines))
    
    word_frequency = dict()
    
    for line in lines:
        for v in line:
            if v in word_frequency:
                word_frequency[v] += 1
            else:
                word_frequency[v] = 1
    
    idx2Id = [item for item, _ in word_frequency.items()]
    id2Idx = {item: idx for idx, item in enumerate(idx2Id)}

    G = nx.Graph()
    test_set = []
    for line in lines:
        G.add_edge(id2Idx[line[0]], id2Idx[line[1]])
        # 采样0.2作为测试集
        if random.random() < 0.1:
            test_set.append([id2Idx[line[0]], id2Idx[line[1]]])

    # test_set均为正样本，
    negative_set = []
    node_num = len(idx2Id)
    for positive in test_set:
        node1, _ = positive
        # 负采样
        neg_node = random.choice(range(node_num))
        # 保证为负样本
        while [idx2Id[node1], idx2Id[neg_node]] in lines:
            neg_node = random.choice(range(node_num))
        negative_set.append([node1, neg_node])
    test_set += negative_set
    print(len(lines))
    
    if not is_weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not is_directed:
        G = G.to_undirected()

    return G, idx2Id, id2Idx, cateDict, test_set
