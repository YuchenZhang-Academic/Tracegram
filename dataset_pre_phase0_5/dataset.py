import os
from scapy.all import *
import torch
import random
import struct
import copy

def pcaplinktype_decoder_class(num):
    if num == 1:
        return scapy.layers.l2.Loopback
    elif num == 2:
        return scapy.layers.l2.Dot3
    elif num == 3:
        return scapy.layers.l2.Ether
    elif num == 4:
        return scapy.layers.ppp
    elif num == 5:
        return scapy.layers.inet6.IPv46
    elif num == 6:
        return scapy.layers.inet6.IPv6
    elif num == 7:
        return scapy.layers.l2.HDLC
    elif num == 8:
        return scapy.layers.dot11
    elif num == 9:
        return scapy.layers.l2.CookedLinux
    elif num == 10:
        return scapy.layers.dot11.PrismHeader
    elif num == 11:
        return scapy.layers.dot11.RadioTap
    elif num == 12:
        return scapy.layers.ppi
    elif num == 13:
        return scapy.layers.dot15d4
    elif num == 14:
        return scapy.layers.bluetooth
    elif num == 15:
        return scapy.layers.l2.DIR_PPP
    elif num == 16:
        return scapy.layers.dot15d4
    elif num == 17:
        return scapy.layers.bluetooth4LE
    elif num == 18:
        return scapy.layers.bluetooth4LE
    elif num == 19:
        return scapy.layers.l2.MPacketPreamble
    elif num == 20:
        return scapy.layers.l2.CookedLinuxV2
    elif num == 21:
        return scapy.layers.inet.IP
    
    return scapy.layers.l2.CookedLinux


def pcaplinktype_encoder(pkt):
    if type(pkt) is scapy.layers.l2.Loopback:
        return  1
    elif type(pkt) is scapy.layers.l2.Dot3:
        return 2
    elif type(pkt) is scapy.layers.l2.Ether:
        return 3
    elif type(pkt) is scapy.layers.ppp:
        return 4
    elif type(pkt) is scapy.layers.inet6.IPv46:
        return 5
    elif type(pkt) is scapy.layers.inet6.IPv6:
        return 6
    # elif type(pkt) is scapy.layers.l2.HDLC:
    #     return 7
    elif type(pkt) is scapy.layers.dot11:
        return 8
    elif type(pkt) is scapy.layers.l2.CookedLinux:
        return 9
    elif type(pkt) is scapy.layers.dot11.PrismHeader:
        return 10
    elif type(pkt) is scapy.layers.dot11.RadioTap:
        return 11
    elif type(pkt) is scapy.layers.ppi:
        return 12
    elif type(pkt) is scapy.layers.dot15d4:
        return 13
    elif type(pkt) is scapy.layers.bluetooth:
        return 14
    # elif type(pkt) is scapy.layers.l2.DIR_PPP:
    #     return 15
    elif type(pkt) is scapy.layers.dot15d4:
        return 16
    elif type(pkt) is scapy.layers.bluetooth4LE:
        return 17
    # elif type(pkt) is scapy.layers.l2.BTLE_RF:
    #     return 18
    elif type(pkt) is scapy.layers.l2.MPacketPreamble:
        return 19
    elif type(pkt) is scapy.layers.l2.CookedLinuxV2:
        return 20
    elif type(pkt) is scapy.layers.inet.IP:
        return 21



class PcapDataset:
    def __init__(self, path_list, max_length):
        self.labels = [os.path.join(path, f) for path in path_list for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        self.label_paths = self.labels
        self.max_length = max_length
        self.data_files = [[] for _ in self.labels]
        
        print('now load path...')
        for i_label in range(len(self.data_files)):
            print('i_label', i_label, 'all', len(self.data_files))
            files = [os.path.join(self.label_paths[i_label], f) for f in os.listdir(self.label_paths[i_label]) if os.path.isfile(os.path.join(self.label_paths[i_label], f))]
            self.data_files[i_label] = files
        print('load finish...')


    def load_pcaps(self, file_path):

        # Read packets from the selected file
        packets = [259] # 259 [cls]
        last_time = None
        d_time = None
        with PcapReader(file_path) as pcap_reader:
            for pkt_data in pcap_reader:

                ts = pkt_data.time
                data = bytes(pkt_data)

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
                vectors.insert(1,pcaplinktype_encoder(pkt_data))
                
                packets.extend(vectors)
                if len(packets) > self.max_length:
                    break
        
        packets = packets[:self.max_length-1] 
        
        end = 257
        packets.append(end)

        return packets

    def seperate_dataset(self, valid_r=0.1, test_r=0.1):
        train_r = 1 - valid_r - test_r

        def subset(data_files, start_r, end_r):
            subdata_files = copy.deepcopy(data_files)
            for i_label in range(len(subdata_files)):
                sample_num = len(subdata_files[i_label])
                start_place = int(sample_num * start_r)
                end_place = int(sample_num * end_r)
                subdata_files[i_label] = subdata_files[i_label][start_place:end_place]

            return subdata_files

        self.trainset = subset(self.data_files, 0, train_r)
        self.validset = subset(self.data_files, train_r, train_r + valid_r)
        self.testset = subset(self.data_files, train_r + valid_r, 1)
    
    def get_trainset(self):
        return self.trainset
    
    def get_validset(self):
        return self.validset
    
    def get_testset(self):
        return self.testset

    def get_batch(self, dataset_files, batch_size=2):
        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            while (True):
                rand_device_id = random.randint(0, len(dataset_files) - 1)
                Y = rand_device_id
                if len(dataset_files[rand_device_id]) <= 0:
                    continue
                rand_sample = random.randint(0, len(dataset_files[rand_device_id]) - 1)
                break

            file_path = dataset_files[rand_device_id][rand_sample]
            batch_x.append(self.load_pcaps(file_path))
            batch_y.append(Y)

        return batch_x, batch_y



if __name__ == "__main__":
    # Example usage
    dataset = PcapDataset(['../../../LLM_traffic_expriments/cls_datasets/test_dataset'], max_length = 5)
    dataset.seperate_dataset(valid_r = 0.1, test_r = 0.1)
    print(dataset.labels)
    print(dataset.trainset)
    print(dataset.testset)













