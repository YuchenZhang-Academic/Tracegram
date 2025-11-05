import torch
from model import get_model
import struct
from dataset import pcaplinktype_encoder
from scapy.all import rdpcap, IP, TCP, UDP, Ether
from config import model_path, dataset_class_num


token_length = 512 #94 * 128
vocab_size = 260
max_len = 94 * 128
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 创建模型实例
model = get_model(num_tokens = vocab_size, max_len = max_len)
model.module.init_cls(class_num = dataset_class_num) # 添加分类用的层
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

def zero_out_packet(packet):
    # 检查并修改 IP 层
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'
    
    # 检查并修改 TCP 层
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0
    
    # 检查并修改 UDP 层
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0
    
    # 检查并修改以太网层
    if Ether in packet:
        packet[Ether].src = '00:00:00:00:00:00'
        packet[Ether].dst = '00:00:00:00:00:00'
    
    return packet


def pcap_tokens(pkgs):
    packets = [259] # 259 [cls]
    last_time = None
    d_time = None

    for pkt in pkgs:
        ts = pkt.time
        pkt = zero_out_packet(pkt)
        data = bytes(pkt)

        if last_time is None:
            d_time = 0
            last_time = ts
        else:
            d_time = ts - last_time
            last_time = ts

        time_encoded = struct.pack('>d', d_time)
        
        vectors = list(time_encoded + data)
        head = 256
        vectors.insert(0,head)
        vectors.insert(1,pcaplinktype_encoder(pkt))
        
        packets.extend(vectors)
        if len(packets) > max_len:
            break
        
    packets = packets[:max_len-1] 
    
    end = 257
    packets.append(end)

    return packets

def token2featrue(tokens):
    global model
    batch_x = [torch.tensor(tokens).long().to(device)]
    with torch.no_grad():
        output = model.module.forward_feature(batch_x)
    output = output.cpu().view(-1).detach().numpy().tolist()
    return output

def LLM_feature(pkgs, show = False):
    tokens = pcap_tokens(pkgs)
    feature = token2featrue(tokens[:token_length])
    if show:
        print('feature', len(feature))
    return feature


def main():
    file = 'G:\\github\\LLM_traffic_expriments\\code\\net3_flow_linear_32\\gen_output_1_2_t0\\125.pcap'
    pkgs = rdpcap(file)
    feature = LLM_feature(pkgs)
    print(feature)


if __name__ == "__main__":
    main()












