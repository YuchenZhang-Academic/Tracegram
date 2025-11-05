# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:03:50 2020

@author: sahua
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy 
from torch.autograd import Variable
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import json
import os
import pytorch_warmup 

train_path = "../dataset_pre_phase1/dataset/captures_IoT-Sentinel_pre_train.json"
valid_path = "../dataset_pre_phase1/dataset/captures_IoT-Sentinel_pre_valid.json"
test_path = "../dataset_pre_phase1/dataset/captures_IoT-Sentinel_pre_test.json"
output_name = './output/20240723_sen/test'

class Net(nn.Module):
    def __init__(self,web_num,emb_num,feature_size):
        super(Net, self).__init__()
        self.web_num = web_num
        self.emb_num = emb_num
        self.feature_size = feature_size

        self.gen_layers()

    def gen_layers(self):
        self.emb2 = nn.Embedding(self.emb_num, self.feature_size)
        #attention net
        self.ss_wk_linear1 = nn.Linear(self.feature_size, self.feature_size)
        self.ss_wv_linear1 = nn.Linear(self.feature_size, self.feature_size)
        self.t_lstm = nn.LSTM(self.feature_size, self.feature_size, 1, bidirectional=True)

        #classifer net
        self.c_conv1 = nn.Conv1d(in_channels=self.emb_num, out_channels=self.emb_num, kernel_size=1)
        self.c_conv2 = nn.Conv1d(in_channels=self.emb_num, out_channels=1, kernel_size=1)
        self.c_linear1 = nn.Linear(self.feature_size, 100)
        self.c_linear2 = nn.Linear(100, 100)
        self.c_linear3 = nn.Linear(100, 50)
        self.c_linear4 = nn.Linear(50, self.web_num)

        self.c_bn1 = nn.BatchNorm1d(self.emb_num)
        self.c_bn2 = nn.BatchNorm1d(self.emb_num)
        self.c_bn3 = nn.BatchNorm1d(100)
        self.c_bn4 = nn.BatchNorm1d(100)

    def trace_feature_generator(self,x):
        if x.shape[0]>0:
            x = x.view(x.shape[0],1,self.feature_size)
            x, (h_n, c_n) = self.t_lstm(x)
            x = x.view(x.shape[0],self.feature_size,-1)
            x = x[:,:,0]  +  x[:,:,1]
        
        #特征向量化
        query = torch.arange(0,self.emb_num).view(-1).long()
        query = query.to(x.device)
        query = self.emb2(query)

        key = self.ss_wk_linear1(x).t()
        value = self.ss_wv_linear1(x)
        
        #softmax(q*kT/sqrt(size)) * v
        attention_a = query.mm(key) / (self.feature_size**0.5)
        attention_soft = F.softmax(attention_a,dim=1).to(x.device)
        attention_x = F.leaky_relu(attention_soft.mm(value))
        attention_x = attention_x.view(-1,self.emb_num,self.feature_size)

        x = attention_x

        final_x = x.view(1,self.emb_num,-1)

        return final_x
    
    def forward2(self,x_in):

        final_x = F.leaky_relu(self.c_conv1(self.c_bn1(x_in)))
        final_x = F.leaky_relu(self.c_conv2(self.c_bn2(final_x)))
        final_x = final_x.view(-1,self.feature_size)

        final_x = F.leaky_relu(self.c_bn3(self.c_linear1(final_x)))
        final_x = F.leaky_relu(self.c_bn4(self.c_linear2(final_x)))
        final_x = F.leaky_relu(self.c_linear3(final_x))
        final_x = self.c_linear4(final_x)
    
        return final_x
    
    def forward1(self, x_in, device):
        x = torch.tensor(x_in).to(device)

        middle_x = self.trace_feature_generator(x)
        return middle_x

    def forward(self, x_in, device):
        x = torch.tensor(x_in).to(device)

        final_x = self.trace_feature_generator(x)
        final_x = self.forward2(final_x)
        
        return final_x


#%%
def train(data_in,model, device,optimizer,batch,warmup_scheduler,lr_scheduler):
    '''
    Parameters
    ----------
    data_in : list
        数据的输入
    batch : int
        每次训练个数
    model : nn.Module
        模型
    device : string
        cpu还是gpu
    epoch : int
        训练轮数,仅用于显示

    Returns
    -------
    一个训练完的分类器
    '''
    #设定参数可变
    model.train()
    
    label_dict = {} # 索引表，方便时候随机使用
    for i in range(len(data_in)):
        if data_in[i][1] not in label_dict.keys():
            label_dict[data_in[i][1]] = []
            label_dict[data_in[i][1]].append(i)
        else:
            label_dict[data_in[i][1]].append(i)
    
    if len(label_dict.keys())<2:
        print('种类不足两类，不需要进行分类....')
        return
    
    sum_loss = 0.5
    
    batch_x_cache = None
    batch_y_cache = None
    
    y_true = []
    y_pred = []

    time0 = time.time()
    for i in range(len(data_in)):
        # 释放变量显存
        torch.cuda.empty_cache()

        rand_i = random.choice(label_dict[random.choice(list(label_dict.keys()))])
        X = data_in[rand_i][0]
        Y = data_in[rand_i][1]
        y_true.append(Y)
        
        # 复制X
        input_X = copy.deepcopy(X)

        # 中途训练数据
        Label = torch.tensor([Y]).to(device)

        middle_result = model.forward1(input_X, device)
            
        if batch_x_cache == None:
            batch_x_cache = middle_result
            batch_y_cache = Label
        else:
            batch_x_cache = torch.cat((batch_x_cache,middle_result),0)
            batch_y_cache = torch.cat((batch_y_cache,Label))
        
        if (i+1) % batch == 0 or ((i == len(data_in) - 1) and (i+1) % batch> 1):  # 按照batch进行处理
            predict = model.forward2(batch_x_cache)
            pred = predict.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            loss = F.cross_entropy(predict,batch_y_cache)

            loss.backward()
            loss_show = loss.item()

            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()
            optimizer.zero_grad()
            
            sum_loss = 0.5*sum_loss + 0.5*loss_show
            
            y_pred.extend(pred.cpu().view(-1).detach().numpy().tolist())
            
            time1 = time.time()
            print('sum_loss',sum_loss,'i/all',i,len(data_in),'cost time:',time1 - time0)
            time0 = time.time()

            # 清空缓存
            batch_x_cache = None
            batch_y_cache = None

    # cut y_true like y_pred
    y_true = y_true[:len(y_pred)]
    
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')

    print('sum_loss',sum_loss,'precision',precision,'recall',recall,'f1',f1score)
    print(confusion_matrix(y_true, y_pred))

    return f1score
        

def test(data_in,model,device,p_text='Test'):
    model.eval() #测试模式
    
    label_dict = {} # 索引表，方便时候随机使用
    for i in range(len(data_in)):
        if data_in[i][1] not in label_dict.keys():
            label_dict[data_in[i][1]] = []
            label_dict[data_in[i][1]].append(i)
        else:
            label_dict[data_in[i][1]].append(i)
    
    # if len(label_dict.keys())<2:
    #     print(p_text,'种类不足两类，不需要进行分类....')
    #     print([d[1] for d in data_in])
    #     return
    
    test_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i in range(len(data_in)):
            X = data_in[i][0]    # 一个数据
            Y = data_in[i][1]    # 一个分类标签
            Label = torch.tensor([Y]).to(device)
            output = model(X,device)
            test_loss += F.cross_entropy(output, Label).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            y_true.append(Y)
            y_pred.append(pred.item())
                 
        
    test_loss /= (2*len(data_in) )       
    
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1score = f1_score(y_true, y_pred, average='macro')

    print()
    print('Average loss',test_loss,'precision',precision,'recall',recall,'f1',f1score)
    print(confusion_matrix(y_true, y_pred))
    print()
    
    return f1score

    
def show_model(model,device):
    x = torch.arange(0*model.emb_num,(0+1)*model.emb_num).view(-1).long()
    x = x.to(device)
    print(model.emb1(x))
    
    return


print('start to readfile...')
with open(train_path,'r') as json_file:
    content = json.load(json_file)
    train_data = content[0]
    label2key_train = content[1]
label_num = len(label2key_train)
print(label2key_train)

with open(valid_path,'r') as json_file:
    content = json.load(json_file)
    valid_data = content[0]
    label2key_valid = content[1]
print(label2key_valid)

with open(test_path,'r') as json_file:
    content = json.load(json_file)
    test_data = content[0]
    label2key_test = content[1]
print(label2key_test)

def replace_label(dataset, label2key):
    for i in range(len(dataset)):
        dataset[i][1] = int(label2key[dataset[i][1]])
    return dataset

train_data = replace_label(train_data, label2key_train)
valid_data = replace_label(valid_data, label2key_valid)
test_data = replace_label(test_data, label2key_test)


# 进行随机打乱
random.seed(1)
random.shuffle(train_data)
random.shuffle(valid_data)
random.shuffle(test_data)

print('读取数据集完成.')
print('test_data[0][0][0][0]',test_data[0][0][0][0])

#%%
'''
主函数入口
'''
def main(output_name):
    #判断是否使用gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    print('use device',device)
    
    # 生成模型
    model = Net(label_num,100,256) 
    model = model.to(device)

    # 文件名字
    para_filename = output_name
    
    # 读取模型参数
    #model.load_state_dict(torch.load(para_filename+'.pt'))
    learning_rate = 1e-3
    
    MAX_EPOCH = 100
    BATCH_SIZE = 16

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    num_steps = int(len(train_data) * MAX_EPOCH / BATCH_SIZE) 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    warmup_scheduler = pytorch_warmup.ExponentialWarmup(optimizer, int(2000/BATCH_SIZE))

    best_f1_score = 0    
    #开始训练
    for epoch in range(MAX_EPOCH):
        print('epoch',epoch)
        train_f1score = train(train_data,model,device,optimizer,BATCH_SIZE,warmup_scheduler,lr_scheduler)
        valid_f1score = test(valid_data,model,device,'valid')

        if valid_f1score>best_f1_score:
            best_f1_score = valid_f1score
            torch.save(model.state_dict(), para_filename +'_best'+'.pt')
        
        # 记录参数数据、训练过程
        with open(para_filename+'.txt',"a") as txt_file:
            txt_file.writelines(str(epoch)+','+str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+'\n')

        #展示训练的模型数据
        #show_model(model,device)
    
        #保存模型
        torch.save(model.state_dict(), para_filename+'.pt')
     
    # 加载最优模型
    model.load_state_dict(torch.load(para_filename+'_best.pt'))

    #开始测试
    train_f1score = test(train_data,model,device,'Train')
    valid_f1score = test(valid_data,model,device,'valid')
    test_f1score = test(test_data,model,device)
    
    with open(para_filename+'.txt',"a") as txt_file:
        txt_file.writelines(str(epoch)+','+str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+','+str(test_f1score)+'\n')


if __name__ == "__main__":
    os.makedirs(os.path.dirname(output_name), exist_ok= True)
    main(output_name)




