
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
import time



# 定义稀疏卷积神经网络
class Trainer():
    def __init__(self, dataset, model, device):
        self.learning_rate = 1e-4
        self.batch_size = 4
        self.max_epoch = 100

        self.dataset = dataset
        self.model = model

        # 选择优化器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.device = device
        self.model = self.model.to(device)


    def train_one_epoch(self, epoch = 0, times = 100):
        self.model.train()
        
        trainset = self.dataset.get_trainset()
        
        y_true = []
        y_pred = []

        running_loss = 0.0
        for i_time in range(times):
            time0 = time.time()

            self.optimizer.zero_grad()

            batch_x, batch_y = self.dataset.get_batch(trainset, batch_size = self.batch_size)
            batch_x = [torch.tensor(x).long().to(self.device) for x in batch_x]
            time1 = time.time()

            output = self.model.module.forward_cls(batch_x)

            Y = torch.LongTensor(batch_y).to(self.device)
            loss = self.criterion(output, Y)
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新模型参数

            running_loss += loss.item()
            time2 = time.time()

            predicted_class = torch.argmax(output, dim=1)
            y_true.extend(batch_y)
            y_pred.extend(predicted_class.cpu().view(-1).detach().numpy().tolist())

            print('epoch',epoch,'step', i_time,'batch loss:',loss.item(),'cost time', time2-time1, time1-time0)


        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')

        print('precision',precision,'recall',recall,'f1',f1score,'acc',acc)
        print(confusion_matrix(y_true, y_pred))

        return f1score
    

    def test(self, mode, times = 100):
        self.model.eval()

        y_true = []
        y_pred = []

        if mode == 'Train':
            testset = self.dataset.get_trainset()
        elif mode == 'Valid':
            testset = self.dataset.get_validset()
        elif mode == 'Test':
            testset = self.dataset.get_testset()

        running_loss = 0.0
        for i_time in range(times):
            batch_x, batch_y = self.dataset.get_batch(testset, batch_size = self.batch_size)
            batch_x = [torch.tensor(x).long().to(self.device) for x in batch_x]
            output = self.model.module.forward_cls(batch_x)

            Y = torch.LongTensor(batch_y).to(self.device)
            loss = self.criterion(output, Y)

            running_loss += loss.item()

            predicted_class = torch.argmax(output, dim=1)
            y_true.extend(batch_y)
            y_pred.extend(predicted_class.cpu().view(-1).detach().numpy().tolist())


        #print('step', i_time,'batch loss:',loss.item())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1score = f1_score(y_true, y_pred, average='macro')

        print('precision',precision,'recall',recall,'f1',f1score,'acc',acc)
        print(confusion_matrix(y_true, y_pred))

        return f1score


    def train(self, folder_path = './output/'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        para_filename = folder_path + 'test'

        with open(para_filename+'.txt',"a") as txt_file:
            txt_file.writelines('epoch'+','+'best_valid_f1score'+','+'train_f1score'+','+'valid_f1score'+'\n')

        best_f1_score = 0
        #开始训练
        for epoch in range(self.max_epoch):
            print("epoch",epoch)
            train_f1score = self.train_one_epoch(epoch)
            valid_f1score = self.test('Valid')
            print('valid_f1score',valid_f1score)

            if valid_f1score>best_f1_score:
                best_f1_score = valid_f1score
                torch.save(self.model.state_dict(), para_filename+'_best'+'.pt')
            
            # 记录参数数据、训练过程
            with open(para_filename+'.txt',"a") as txt_file:
                txt_file.writelines(str(epoch)+','+str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+'\n')

            #保存模型
            torch.save(self.model.state_dict(), para_filename+'.pt')

            #学习率衰减
            self.learning_rate *= 0.98

        # 读取模型参数
        self.model.load_state_dict(torch.load(para_filename+'_best'+'.pt',map_location=self.device))

        #开始测试
        train_f1score = self.test('Train')
        valid_f1score = self.test('Valid')
        test_f1score = self.test('Test')
        
        with open(para_filename+'.txt',"a") as txt_file:
            txt_file.writelines(str(best_f1_score)+','+str(train_f1score)+','+str(valid_f1score)+','+str(test_f1score)+'\n')










