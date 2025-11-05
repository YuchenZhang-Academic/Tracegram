import torch
from model import get_model

from dataset import PcapDataset
from trainer import Trainer

def main():
    token_length = 94 * 128
    vocab_size = 260
    max_len = 94 * 128
    
    model_path = './output/20240608/test.pt'
    #model_path = '/qj/LLM/code/net3_flow_linear_32/output/run_v2/test_234.pt'#'./model_files/test_9.pt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    '''
    dataset_path = ['/qj/LLM/cls_datasets/Cross-Platform-pre5-android',
                     '/qj/LLM/cls_datasets/Cross-Platform-pre5-ios',
                     '/qj/LLM/cls_datasets/cstnet-tls/cstnet-tls 1.3',
                     '/qj/LLM/cls_datasets/NETGPT-task2_step3',
                     '/qj/LLM/cls_datasets/USTC_step3',
                     ]
    '''
    dataset_path = ['../dataset_pre_phase0_5/dataset/5pmnkshffm-3/flow/train',
                    '../dataset_pre_phase0_5/dataset/5pmnkshffm-3/flow/valid']
    
    output_path = './output/pre_5p/'

    # 加载数据集
    print('start load dataset')
    dataset = PcapDataset(dataset_path, max_length = token_length)
    dataset.seperate_dataset(valid_r = 0.1, test_r = 0.01)

    # 创建模型实例
    model = get_model(num_tokens = vocab_size, max_len = max_len)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.module.init_cls(class_num = len(dataset.labels))
    #model.module.init_cls(class_num = len(dataset.labels)) # 添加分类用的层


    mytrainer = Trainer(dataset, model, device)
    mytrainer.train(output_path)

    return 

if __name__ == "__main__":
    main()










