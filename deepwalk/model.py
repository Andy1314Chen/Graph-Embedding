import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

cuda_gpu = torch.cuda.is_available()

class SkipGramModel(nn.Module):
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size                    # vocab size 
        self.emb_dimension = emb_dimension          # embedding size
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.log_sigmoid = nn.LogSigmoid()
        self.init_emb()

    def init_emb(self):
        initrange = (2.0 / (self.emb_size + self.emb_dimension)) ** 0.5 # 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, target_word, context_word, neg_word):
        # [B, emb_size]
        emb_u = self.u_embeddings(target_word)
        emb_v = self.v_embeddings(context_word)
        # [B]
        positive = self.log_sigmoid(torch.sum(emb_u * emb_v, dim=1)).squeeze()

        # [B, neg_num, emb_size]
        emb_neg = self.v_embeddings(neg_word)
        # [B, neg_num, emb_size] * [B, emb_size, 1] = [B, neg_num, 1]
        # [B, neg_num]
        negative = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze(2)
        # [B]
        negative = self.log_sigmoid(-torch.sum(negative, dim=1)).squeeze()

        loss = positive + negative
        return -loss.mean()

    def save_embedding(self, file_name):
        if cuda_gpu:
            embedding = self.u_embeddings.cpu().weight.data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        with open(file_name, "w", encoding="UTF-8") as fout:
            # fout.write("%d %d\n" % (len(id2word), self.emb_dimension))
            for wid in range(len(embedding)):
                e = embedding[wid]
                e = '\t'.join(map(lambda x: str(x), e))
                fout.write("%s\n" % e)

