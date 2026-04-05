import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import torchvision 
import copy
import json
import math
import string 


class SimpleVocab(object):

    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.word2id = {}
        self.wordcount = {}
        self.word2id['<UNK>'] = 0
        self.word2id['<AND>'] = 1
        self.word2id['<BOS>'] = 2
        self.word2id['<EOS>'] = 3
        self.wordcount['<UNK>'] = 9e9
        self.wordcount['<AND>'] = 9e9
        self.wordcount['<BOS>'] = 9e9
        self.wordcount['<EOS>'] = 9e9

    def tokenize_text(self, text):   # 把字符串去掉符号变成单词的list
        text = text.encode('ascii', 'ignore').decode('ascii')
        trans = str.maketrans({key: None for key in string.punctuation})
        tokens = str(text).lower().translate(trans).strip().split()
        return tokens

    def add_text_to_vocab(self, text):   # 每个单词建立word2id 以及 wordcount 的字典
        tokens = self.tokenize_text(text)
        for token in tokens:
            if not token in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def threshold_rare_words(self, wordcount_threshold=3):  # 去掉出现的单词数量小于5的词
        for w in self.word2id:
            if self.wordcount[w] < wordcount_threshold:
                self.word2id[w] = 0

    def encode_text(self, text):  # 把句子转变成数字
        tokens = self.tokenize_text(text)
        # dict.get(key, default=None)
        x = [self.word2id.get(t, 0) for t in tokens]
        return x

    def get_size(self):  # 单词数量
        return len(self.word2id)


class TextLSTMModel(torch.nn.Module):
    def __init__(self, texts_to_build_vocab=None, word_embed_dim=512, lstm_hidden_dim=256, outfit_threshold=10):
        super().__init__()
        self.vocab = SimpleVocab()
        for text in texts_to_build_vocab:
            self.vocab.add_text_to_vocab(text)
        vocab_size = self.vocab.get_size()
        self.word_embed_dim = word_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.embedding_layer = torch.nn.Embedding(vocab_size, word_embed_dim)
        self.lstm = torch.nn.LSTM(word_embed_dim, lstm_hidden_dim)
        self.lstm_fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        )
        self.outfit_threshold = outfit_threshold

    def forward(self, x):
        """ input x: list of strings"""
        if type(x) is list and type(x[0]) is list:
            if type(x[0][0]) is str:
                for index, data in enumerate(x):
                    x[index] = [self.vocab.encode_text(text) for text in data]

        return self.forward_encoded_texts(x)


    def forward_encoded_texts(self, texts):
        # to tensor
        batch_size = len(texts)
        lengths = []                         # [[3,5,10],[3,6,3,3],...]  batch_size, outfit, .. 数字代表单个item的文本的长度，里边那层list里元素的个数表示outfit中item的个数
        for index, data in enumerate(texts):
            lengths.append([len(t) for t in data])    
        
        max_length = np.max([np.max(t) for t in lengths])
        max_outfit = np.max([len(t) for t in lengths])
        max_outfit = min(max_outfit, self.outfit_threshold)

        tensor_texts = torch.zeros((batch_size, max_length, max_outfit)).long()
        #adj = torch.zeros((batch_size, max_outfit, max_outfit))
 
        for index, data in enumerate(texts):
            length_index = lengths[index]
            itexts = torch.zeros((max_length, max_outfit)).long()
            for i in range(min(len(length_index), self.outfit_threshold)):
                itexts[:length_index[i], i] = torch.tensor(data[i])
   
            tensor_texts[index] = itexts
        tensor_texts = torch.autograd.Variable(tensor_texts.permute(2,1,0).contiguous()).cuda()

        lstm_outs = torch.zeros((max_outfit, batch_size, self.lstm_hidden_dim)).cuda()
        for index in range(max_outfit):
            itexts = tensor_texts[index]
            etexts = self.embedding_layer(itexts)

            lstm_output,_ = self.forward_lstm_(etexts)   # length, batch, dim
   
            text_features = []
            for i in range(batch_size):
                dim0_index = lengths[i][index] -1 if index < len(lengths[i]) else max_length -1
                text_features.append(lstm_output[dim0_index, i, :])   
            text_features = torch.stack(text_features)  # batch, dim
            text_features = self.lstm_fc(text_features)
            lstm_outs[index] = text_features
        lstm_outs = lstm_outs.permute(1,0,2).contiguous()

        return lstm_outs

    def forward_lstm_(self, etexts):
        batch_size = etexts.shape[1]
        first_hidden = (torch.zeros(1, batch_size, self.lstm_hidden_dim),
                        torch.zeros(1, batch_size, self.lstm_hidden_dim))
        first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden


class Image_net(nn.Module):
    def __init__(self, texts_to_build_vocab, embedding_size=256, outfit_threshold=10):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.backbone.fc = torch.nn.Linear(512, embedding_size)
        self.text_backbone = TextLSTMModel(texts_to_build_vocab=texts_to_build_vocab, word_embed_dim=512, lstm_hidden_dim=embedding_size)

        self.hidden_dim = embedding_size
        self.outfit_threshold = outfit_threshold

        self.text_mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)) for i in range(2))

        self.img_mlps = nn.ModuleList(nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)) for i in range(2))

        self.ego = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )
        self.mess = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
        )
        self.classify = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim // 2, 2)
        )

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = 0.1

    def extract_text_feature(self, texts):
        text_feature = self.text_backbone(texts)
        return text_feature 

    def extract_img_feature(self, img):
        padding = torch.zeros_like(img[0][0])
        batch_size = len(img)
        max_outfit = max([len(t) for t in img])
        max_outfit = min(max_outfit, self.outfit_threshold)

        adj = torch.zeros((batch_size, max_outfit, max_outfit))
        y = torch.zeros((batch_size, max_outfit, 3, 224, 224))

        for index, t in enumerate(img):
            adj_matrix = torch.ones(max_outfit, max_outfit) - torch.eye(max_outfit)
            #y[index][:len(t)] = torch.stack(t)
            y[index][:min(len(t), self.outfit_threshold)] = torch.stack(t[:min(len(t), self.outfit_threshold)])
            if len(t) < max_outfit:
                adj_matrix[len(t)-max_outfit:,:] = 0
                adj_matrix[:,len(t)-max_outfit:] = 0

            adj[index] = adj_matrix

        y = y.cuda().permute(1,0,2,3,4).contiguous()

        feature = torch.zeros((max_outfit, batch_size, self.hidden_dim)).cuda()
        for index in range(max_outfit):
            feature[index] = self.backbone(y[index])
        feature = feature.permute(1,0,2).contiguous()    # max_out, batch_size, dim  -> batch_size, nodes, dim
        
        adj = adj.cuda()

        return feature, adj                     # batch_size, nodes, dim   ;  batch_size, outfitnum, outfitnum
    
    def compute_complementary_feature(self, img_feature, text_feature, adj):
        batch_size, outfitnum, dim = img_feature.size()
        adj_mask = torch.sum(adj,dim=1)   # batch_size, outfitnum  padding 0
        adj_mask = torch.where(adj_mask>0, adj_mask/adj_mask, adj_mask)

        ortho_text = self.text_mlps[0](text_feature)
        consist_text = self.text_mlps[1](text_feature)

        ortho_img = self.img_mlps[0](img_feature)
        consist_img = self.img_mlps[1](img_feature)
        # consist loss
        img_text_consist = torch.matmul(F.normalize(consist_img, dim=-1), F.normalize(consist_text.permute(0,2,1), dim=-1))  # batch_size, outfitnum, outfitnum
        I = torch.eye(outfitnum).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        loss_consist = F.mse_loss(img_text_consist * adj_mask.unsqueeze(-1).repeat(1,1,outfitnum), I * adj_mask.unsqueeze(-1).repeat(1,1,outfitnum))

        # ortho loss
        img_text_ortho = torch.matmul(F.normalize(ortho_img, dim=-1), F.normalize(ortho_text.permute(0,2,1), dim=-1))
        img_text_ortho = torch.diagonal(img_text_ortho, dim1=-2, dim2=-1)
        zeros = torch.zeros_like(img_text_ortho).cuda()     # batch_size, outfitnum
        loss_ortho = F.mse_loss(img_text_ortho * adj_mask, zeros)

        return loss_consist, loss_ortho, ortho_text
    
    
    def _get_outfit_graph_feat(self, adj, img_embedding, masked_embedding, mask):

        adj_origin = adj
        batch_size, outfit_num, mask_k, dim = masked_embedding.size()

        masked_embedding_in_chunks = masked_embedding.repeat(1, 1, outfit_num, 1).view(batch_size, outfit_num * outfit_num, mask_k, dim)
        masked_embedding_alternating = masked_embedding.repeat(1, outfit_num, 1, 1)   # batch_size, outfit_num * outfit_num, mask_k, dim
        relation = masked_embedding_in_chunks * masked_embedding_alternating  # batch_size, outfit_num * outfit_num, mask_k, dim 
        relation = torch.sum(relation, dim=2)   # batch_size, outfit_num * outfit_num, dim 
        adj = adj.view(-1, outfit_num * outfit_num, 1).repeat(1, 1, self.hidden_dim)
        relation = relation * adj             
        relation = relation.view(-1, outfit_num, outfit_num, self.hidden_dim)

        adj = torch.sum(adj_origin,dim=1)   # batch_size, outfitnum  padding 0
        seq_len = torch.where(adj>0, adj/adj, adj).unsqueeze(-1)   # batch_size, outfitnum, 1

        relation = torch.sum(relation, dim=2)

        relation = torch.div(relation, torch.sum(seq_len, dim=1).unsqueeze(1))

        com_mess = self.mess(relation)
        ego_mess = self.ego(img_embedding)

        new_code = (ego_mess + com_mess).view(-1, outfit_num, self.hidden_dim)  # batch_size, outfit_num, dim
        new_code = new_code * seq_len

        return new_code
    

    def _compatibility_score(self, ilatents, adj):
        adj = torch.sum(adj,dim=1)   # batch_size, outfitnum  padding 0
        mask = torch.where(adj>0, adj/adj, adj).unsqueeze(-1)  
        score = self.classify(ilatents)  # batch_size, outfitnum, 2
        score = torch.sum(score*mask, dim=1)  # batch_size, 2
        score = torch.div(score, torch.sum(mask, dim=1))
        return score

    def forward(self, img_origin, text_origin):
        img = copy.deepcopy(img_origin)
        text = copy.deepcopy(text_origin)

        img, adj = self.extract_img_feature(img)     # img: batch_size, nodes, dim    adj: batch_size, nodes, nodes
        text = self.extract_text_feature(text)

        img_detach = img.detach()
        loss_consist, loss_ortho, ortho_text = self.compute_complementary_feature(img_detach, text, adj)

        img = img + ortho_text

        adj_mask = torch.sum(adj,dim=1)   # batch_size, outfitnum  padding 0
        mask_dim = torch.where(adj_mask>0, adj_mask/adj_mask, adj_mask).unsqueeze(-1).repeat(1, 1, self.hidden_dim)
        img = img * mask_dim   # batch_size, outfitnum, dim
        masked_embedding = img.unsqueeze(2)  # (batch_size, outfitnum, 1, dim)

        I = self._get_outfit_graph_feat(adj, img, masked_embedding, mask_dim)
        score = self._compatibility_score(I, adj)

        return score, loss_consist, loss_ortho