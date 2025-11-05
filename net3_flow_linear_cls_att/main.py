import torch
from model import get_model

from dataset import PcapDataset
from trainer import Trainer

def main():
    token_length = 256
    vocab_size = 260
    max_len = 2*128
    model_path = './model_files/test_9.pt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset_path = '../../cls_datasets/test_dataset'
    output_path = './output/'

    # 加载数据集
    print('start load dataset')
    dataset = PcapDataset(dataset_path, max_length = token_length)
    dataset.seperate_dataset(valid_r = 0.1, test_r = 0.1)

    # 创建模型实例
    model = get_model(num_tokens = vocab_size, max_len = max_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.module.init_cls(class_num = len(dataset.labels)) # 添加分类用的层

    mytrainer = Trainer(dataset, model, device)
    mytrainer.train(output_path)

    return 

if __name__ == "__main__":
    main()










