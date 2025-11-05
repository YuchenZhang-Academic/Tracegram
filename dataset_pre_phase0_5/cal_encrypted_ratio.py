import os
import dpkt
import multiprocessing
from tqdm import tqdm

def is_encrypted_packet(packet):
    eth = dpkt.ethernet.Ethernet(packet)
    if isinstance(eth.data, dpkt.ip.IP):
        ip = eth.data
        if isinstance(ip.data, dpkt.tcp.TCP):
            tcp = ip.data
            if tcp.dport == 443 or tcp.sport == 443:
                return True
    return False

def process_file(filepath):
    total_packets = 0
    encrypted_packets = 0

    with open(filepath, 'rb') as f:
        try:
            try:
                pcap = dpkt.pcap.Reader(f)  # 先按.pcap格式解析，若解析不了，则按pcapng格式解析
            except:
                f.seek(0,0)
                pcap = dpkt.pcapng.Reader(f)

            for _, packet in pcap:
                total_packets += 1
                if is_encrypted_packet(packet):
                    encrypted_packets += 1
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return 0, 0

    return encrypted_packets, total_packets

def get_pcap_files(directory):
    pcap_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng'):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def main(directory):
    pcap_files = get_pcap_files(directory)
    
    total_encrypted_packets = 0
    total_packets = 0

    with multiprocessing.Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(process_file, pcap_files), total=len(pcap_files)))

    for encrypted, total in results:
        total_encrypted_packets += encrypted
        total_packets += total

    if total_packets == 0:
        print("No packets found.")
    else:
        encrypted_ratio = total_encrypted_packets / total_packets
        print(f"Encrypted packets ratio: {encrypted_ratio:.2%}")


if __name__ == "__main__":
    directory = './org_dataset/5pmnkshffm-3'
    main(directory)

    directory = './org_dataset/and50'
    main(directory)

    directory = './org_dataset/captures_IoT-Sentinel'
    main(directory)

    directory = './org_dataset/ETF IoT'
    main(directory)

