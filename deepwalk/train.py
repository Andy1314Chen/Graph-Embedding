# -*- coding: utf-8 -*-
import os
import csv
import torch
import torch.nn as nn 

from tqdm import tqdm 
from input_fn import InputData
from model import SkipGramModel
from graph import Graph
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

cuda_gpu = torch.cuda.is_available()
print("cuda_gpu:", cuda_gpu)
# print("current gpu:", torch.cuda.current_device())

class Word2Vec(object):
    def __init__(self,output_file_name,
            walks = [],
            emb_dimension=100,
            batch_size=64,
            window_size=5,
            epochs=5,
            negative_num=5):
        print("Load data...")
        self.data = InputData(window_size, batch_size, walks)
        self.output_file_name = output_file_name
        self.emb_dimension = emb_dimension
        self.epochs = epochs
        self.negative_num = negative_num
        self.batch_size = batch_size
        self.vocab_size = self.data.vocab_size
        self.model = SkipGramModel(self.vocab_size, self.emb_dimension)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)

        if cuda_gpu:
            self.model = self.model.cuda()

    def train_model(self):
        for _ in tqdm(range(self.epochs)):
            step = 0
            avg_loss = 0
            for pos_pairs in self.data.data_iter:

                target_word = pos_pairs[0][:,0]
                context_word = pos_pairs[0][:,1]
                neg_word = self.data.get_negative_sample(pos_pairs[0], 3)

                if cuda_gpu:
                    target_word = torch.tensor(target_word, dtype=torch.long).cuda()
                    context_word= torch.tensor(context_word, dtype=torch.long).cuda()
                    neg_word = torch.tensor(neg_word, dtype=torch.long).cuda()
                    loss = self.model(target_word, context_word, neg_word).cuda()

                else:
                    target_word = torch.tensor(target_word, dtype=torch.long)
                    context_word= torch.tensor(context_word, dtype=torch.long)
                    neg_word = torch.tensor(neg_word, dtype=torch.long)

                    loss = self.model(target_word, context_word, neg_word)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if cuda_gpu:
                    avg_loss += loss.cpu().item()
                else:
                    # print(loss.item())
                    avg_loss += loss.item()
                step += 1
                if step % 2000 == 0 and step > 0:
                    avg_loss /= 2000
                    print("Average loss at step ", step, ": ", avg_loss)
                    avg_loss = 0

        self.model.save_embedding(self.output_file_name)
        print("~ done.")
    
if __name__ == "__main__":
    # read graph
    nx_G, idx2Id, id2Idx, cateDict, test_set = read_graph("data/cora.cites")

    # random walk
    G = Graph(nx_G, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(10, 20)
    print(len(walks))

    # trian model
    model = Word2Vec("result/embedding.tsv", walks)
    model.train_model()
    print("~ done.")

    # save result 可视化 http://projector.tensorflow.org/
    item_meta, cateList, y = [], [], []
    for i, id in enumerate(idx2Id):
        cate = cateDict.get(id, "unknown")
        item_meta.append(str(id) + "\t" + cate)
        if cate in cateList:
            y.append(cateList.index(cate))
        else:
            y.append(len(cateList))
            cateList.append(cate)
    
    item_meta = ["item_id" + "\t" + "item_cate"] + item_meta
    
    with open("result/item_meta.tsv", "w", newline='\n', encoding="utf-8") as f:
        tsv_output = csv.writer(f, delimiter='\n')
        tsv_output.writerow(item_meta)

    with open("result/embedding.tsv", "r") as f:
        lines = f.readlines()
        lines = list(map(lambda line: [float(x) for x in line.strip().split("\t")], lines))
    
    X = np.array(lines)
    y = np.array(y)

    # 将Embedding作为特征，进行多分类实验
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=163)
    clf = LogisticRegression(random_state=163, solver='lbfgs',
                        multi_class='multinomial').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("macro f1: ", f1_score(y_test, y_pred, average="macro"))
    print("micro f1: ", f1_score(y_test, y_pred, average="micro"))
    """
    macro f1:  0.8289254798238617
    micro f1:  0.8321033210332104
    """