# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 19:29:59 2021

@author: sahua
"""

import os
from scapy.all import *
import time
import json

#import dpkt
import socket
from LLM_feature import LLM_feature
from tqdm import tqdm

#%%


def calculate_total_packet_size(pkt_list):
    """
    计算所有数据包的总大小

    :param pkt_list: 数据包列表
    :return: 总大小（字节）
    """
    total_size = 0
    for pkt in pkt_list:
        total_size += len(pkt)
    return total_size

# 遍历文件夹
def walkFile(file):
    for root, dirs, files in os.walk(file):
        # 遍历所有的文件夹
        for d in dirs:
            filename = os.path.join(root, d)

            file_path_dcit[d] = []
            for root2, dirs2, files2 in os.walk(filename):
                for d2 in files2:
                    filename2 = os.path.join(root2, d2)
                    file_path_dcit[d].append(filename2)
                    

def fast_pkt_info(i_pkt):
    try:
        if i_pkt['IP'].proto == 6: #TCP
            src = i_pkt['IP'].src
            dst = i_pkt['IP'].dst
            sport = i_pkt['TCP'].sport
            dport = i_pkt['TCP'].dport
            
        elif  i_pkt['IP'].proto == 17: #UDP
            src = i_pkt['IP'].src
            dst = i_pkt['IP'].dst
            sport = i_pkt['UDP'].sport
            dport = i_pkt['UDP'].dport
        else:
            return []
        
        transf_type = i_pkt['IP'].proto
        
        if src > dst:
            direction = 1
            unit = (transf_type,src,dst,sport,dport)
        else:
            direction = -1
            unit = (transf_type,dst,src,dport,sport)
            
        return [unit, i_pkt]
        
    except:
        return []
        

def fast_read_pcap(pbar, input_file,label,split_second = 5, min_size = 0):
    return_list = []
    flows = {}
    
    first_start_flag = True
    start_time = 0
    pkt_id = 0  # 初始序列为0
    
    with PcapReader(input_file) as pcap_reader:

        for pkt in pcap_reader:  # 遍历pcap数据
            pkt_id += 1
            if first_start_flag == True:
                first_start_flag = False
                start_time = pkt.time
                pktinfo1 = fast_pkt_info(pkt)
                if len(pktinfo1) >0 :
                    unit = pktinfo1[0]
                    if unit not in flows.keys():
                        flows[unit] = []
                    flows[unit].append(pkt)
                continue
            
            if pkt.time - start_time <= split_second:
                pktinfo1 = fast_pkt_info(pkt)
                if len(pktinfo1) >0 :
                    unit = pktinfo1[0]
                    if unit not in flows.keys():
                        flows[unit] = []
                    flows[unit].append(pkt)
            else:
                # 进行分流，处理特征
                list_values = [i for i in flows.values()]
                #print('start to cal LLM_feature..')
                flattened_list = [item for sublist in list_values for item in sublist]
                if len(list_values)>0 and calculate_total_packet_size(flattened_list) >= min_size:
                    list_values = [LLM_feature(flow) for flow in list_values]
                    pbar.update(len(list_values))
                    return_list.append([list_values,label])
                
                #后处理
                start_time = start_time + split_second
                flows = {}
                
    #后处理
    list_values = [i for i in flows.values()]
    #print('start to cal LLM_feature..')
    flattened_list = [item for sublist in list_values for item in sublist]
    if len(list_values)>0 and calculate_total_packet_size(flattened_list) >= min_size:
        pbar.update(len(list_values))
        list_values = [LLM_feature(flow) for flow in list_values]
        return_list.append([list_values,label])

    return return_list


def main(file_path, output_file, split_second, min_size):
    global file_path_dcit, final_labeled_list, label2key
    file_path_dcit = {}
    final_labeled_list = []
    label2key = []


    walkFile(file_path)
    print(file_path_dcit)
    label = 0
    total_files = sum(len(files) for files in file_path_dcit.values())
    pbar = tqdm(total=total_files, desc="Processing files")

    for key in file_path_dcit.keys():
        sample_num =0 
        for pcapfile in file_path_dcit[key]:
            if not (pcapfile.endswith('.pcap') or pcapfile.endswith('.pcapng') ):
                continue

            return_list = fast_read_pcap(pbar, pcapfile,label, split_second, min_size)

        



if __name__ == '__main__':
    file_path = "../dataset_pre_phase0_5/dataset/5pmnkshffm-3_remove_payload/trace/train"
    output_file = "./dataset/dataset_5p_remove_payload_nopre2_train.json"
    main(file_path, output_file, split_second = 5, min_size = 3*1024)
    file_path = "../dataset_pre_phase0_5/dataset/5pmnkshffm-3_remove_payload/trace/valid"
    output_file = "./dataset/dataset_5p_remove_payload_nopre2_valid.json"
    main(file_path, output_file, split_second = 5, min_size = 3*1024)
    file_path = "../dataset_pre_phase0_5/dataset/5pmnkshffm-3_remove_payload/trace/test"
    output_file = "./dataset/dataset_5p_remove_payload_nopre2_test.json"
    main(file_path, output_file, split_second = 5, min_size = 3*1024)
    
    file_path = "../dataset_pre_phase0_5/dataset/ETF IoT_remove_payload/trace/train"
    output_file = "./dataset/ETF_IoT_remove_payload_nopre2_train.json"
    main(file_path, output_file, split_second = 60, min_size = 0)
    file_path = "../dataset_pre_phase0_5/dataset/ETF IoT_remove_payload/trace/valid"
    output_file = "./dataset/ETF_IoT_remove_payload_nopre2_valid.json"
    main(file_path, output_file, split_second = 60, min_size = 0)
    file_path = "../dataset_pre_phase0_5/dataset/ETF IoT_remove_payload/trace/test"
    output_file = "./dataset/ETF_IoT_remove_payload_nopre2_test.json"
    main(file_path, output_file, split_second = 60, min_size = 0)

    file_path = "../dataset_pre_phase0_5/dataset/captures_IoT-Sentinel_remove_payload/trace/train"
    output_file = "./dataset/captures_IoT-Sentinel_remove_payload_nopre2_train.json"
    main(file_path, output_file, split_second = 1e5, min_size = 0)
    file_path = "../dataset_pre_phase0_5/dataset/captures_IoT-Sentinel_remove_payload/trace/valid"
    output_file = "./dataset/captures_IoT-Sentinel_remove_payload_nopre2_valid.json"
    main(file_path, output_file, split_second = 1e5, min_size = 0)
    file_path = "../dataset_pre_phase0_5/dataset/captures_IoT-Sentinel_remove_payload/trace/test"
    output_file = "./dataset/captures_IoT-Sentinel_remove_payload_nopre2_test.json"
    main(file_path, output_file, split_second = 1e5, min_size = 0)

    file_path = "../dataset_pre_phase0_5/dataset/and50_remove_payload/trace/train"
    output_file = "./dataset/and50_remove_payload_nopre2_train.json"
    main(file_path, output_file, split_second = 1e5, min_size = 0)
    file_path = "../dataset_pre_phase0_5/dataset/and50_remove_payload/trace/valid"
    output_file = "./dataset/and50_remove_payload_nopre2_valid.json"
    main(file_path, output_file, split_second = 1e5, min_size = 0)
    file_path = "../dataset_pre_phase0_5/dataset/and50_remove_payload/trace/test"
    output_file = "./dataset/and50_remove_payload_nopre2_test.json"
    main(file_path, output_file, split_second = 1e5, min_size = 0)



