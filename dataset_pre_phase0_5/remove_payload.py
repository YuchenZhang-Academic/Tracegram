import os
import shutil
from scapy.all import rdpcap, wrpcap, TCP, UDP
from tqdm import tqdm

def remove_application_layer(packet):
    if TCP in packet:
        del packet[TCP].payload
    elif UDP in packet:
        del packet[UDP].payload

def copy_pcap_files_with_structure(src_dir, dst_dir):
    # Get the list of all pcap files to process
    pcap_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, file))
    
    # Iterate over the pcap files with a progress bar
    for src_file_path in tqdm(pcap_files, desc="Processing PCAP files"):
        relative_path = os.path.relpath(os.path.dirname(src_file_path), src_dir)
        dst_file_dir = os.path.join(dst_dir, relative_path)
        dst_file_path = os.path.join(dst_file_dir, os.path.basename(src_file_path))
        
        # Create destination directory if it does not exist
        os.makedirs(dst_file_dir, exist_ok=True)
        
        # Read packets, remove application layer data, and write to the destination file
        packets = rdpcap(src_file_path)
        for pkt in packets:
            remove_application_layer(pkt)
        
        wrpcap(dst_file_path, packets)

# Usage example:
# src_folder = './dataset/and50'
# dst_folder = './dataset/and50_remove_payload'
# copy_pcap_files_with_structure(src_folder, dst_folder)

src_folder = './dataset/5pmnkshffm-3'
dst_folder = './dataset/5pmnkshffm-3_remove_payload'
copy_pcap_files_with_structure(src_folder, dst_folder)

src_folder = './dataset/captures_IoT-Sentinel'
dst_folder = './dataset/captures_IoT-Sentinel_remove_payload'
copy_pcap_files_with_structure(src_folder, dst_folder)

src_folder = './dataset/ETF IoT'
dst_folder = './dataset/ETF IoT_remove_payload'
copy_pcap_files_with_structure(src_folder, dst_folder)


