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
import random

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
        

def fast_read_pcap(input_file,label,split_second = 5, min_size = 0):
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
                flattened_list = [item for sublist in list_values for item in sublist]
                if len(list_values)>0 and calculate_total_packet_size(flattened_list) >= min_size:
                    #for flow in list_values:
                    return_list.append([list_values,label])
                
                #后处理
                start_time = start_time + split_second
                flows = {}
                
    #后处理
    list_values = [i for i in flows.values()]
    flattened_list = [item for sublist in list_values for item in sublist]
    if len(list_values)>0 and calculate_total_packet_size(flattened_list) >= min_size:
        #for flow in list_values:
        return_list.append([list_values,label])

    return return_list

def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def sort_packets_by_time(packet_list):
    # 使用sorted函数和lambda表达式按照time属性排序
    sorted_packets = sorted(packet_list, key=lambda pkt: pkt.time)
    return sorted_packets

def save_packets(packets_list, output_path, train_rate, valid_rate):
    global file_num, trace_file_num
    for flow_list, filename in packets_list:
        flow_dataset_path =  os.path.join(output_path, 'flow')
        trace_dataset_path =  os.path.join(output_path, 'trace')

        p = random.random()
        if p < train_rate:
            flow_dataset_path = os.path.join(flow_dataset_path, 'train')
            trace_dataset_path = os.path.join(trace_dataset_path, 'train')
        elif p < train_rate + valid_rate:
            flow_dataset_path = os.path.join(flow_dataset_path, 'valid')
            trace_dataset_path = os.path.join(trace_dataset_path, 'valid')
        else:
            flow_dataset_path = os.path.join(flow_dataset_path, 'test')
            trace_dataset_path = os.path.join(trace_dataset_path, 'test')

        flow_dataset_path = os.path.join(flow_dataset_path, str(filename))
        trace_dataset_path = os.path.join(trace_dataset_path, str(filename))

        # 检查输出路径是否存在，如果不存在则创建
        if not os.path.exists(flow_dataset_path):
            os.makedirs(flow_dataset_path)
        if not os.path.exists(trace_dataset_path):
            os.makedirs(trace_dataset_path)

        # 存储flow数据集
        pkg_num = 0
        for flow in flow_list:
            file_path = os.path.join(flow_dataset_path, str(file_num)+ '.pcap')
            file_num += 1
            pkg_num += len(flow)
            wrpcap(file_path, flow)
            print(f"flow数据包已保存到: {file_path}")
        print('保存flow中数据包个数:', pkg_num)

        # 存储trace数据集
        
        pkgs = flatten(flow_list)
        pkgs = sort_packets_by_time(pkgs)
        file_path = os.path.join(trace_dataset_path, str(file_num)+ '.pcap')
        trace_file_num += 1
        wrpcap(file_path, pkgs)
        print(f"trace数据包已保存到: {file_path}")
        print('保存flow中数据包个数:', len(pkgs))
        


def main(file_path, output_path, split_second, min_size, train_rate = 0.8, valid_rate = 0.1):
    global file_path_dcit, final_labeled_list, label2key, file_num, trace_file_num
    file_path_dcit = {}
    final_labeled_list = []
    label2key = []

    file_num = 0
    trace_file_num = 0

    walkFile(file_path)
    print(file_path_dcit)
    label = 0
    for key in file_path_dcit.keys():
        sample_num =0 
        for pcapfile in file_path_dcit[key]:
            if not (pcapfile.endswith('.pcap') or pcapfile.endswith('.pcapng') ):
                continue
            print(pcapfile)
            return_list = fast_read_pcap(pcapfile,label, split_second, min_size)
            print(len(return_list))
            sample_num += len(return_list)

            # 写文件
            save_packets(return_list, output_path, train_rate, valid_rate)
        
        print(key,'sample_num',sample_num)
        
        label += 1
        label2key.append(key)
            

if __name__ == '__main__':

    file_path = "./org_dataset/5pmnkshffm-3/"
    output_path = "./dataset/5pmnkshffm-3/"
    main(file_path, output_path, split_second = 5, min_size = 3*1024, train_rate = 0.8, valid_rate = 0.1)

    file_path = "./org_dataset/ETF IoT/"
    output_path = "./dataset/ETF IoT/"
    main(file_path, output_path, split_second = 60, min_size = 0, train_rate = 0.8, valid_rate = 0.1)

    file_path = "./org_dataset/captures_IoT-Sentinel/"
    output_path = "./dataset/captures_IoT-Sentinel/"
    main(file_path, output_path, split_second = 1e5, min_size = 0, train_rate = 0.8, valid_rate = 0.1)

    file_path = "./org_dataset/and50/"
    output_path = "./dataset/and50/"
    main(file_path, output_path, split_second = 1e5, min_size = 0, train_rate = 0.8, valid_rate = 0.1)
    
    
    