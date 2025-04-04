from opencc import OpenCC
from tqdm import tqdm
import sys
import os

current_file_path = os.path.abspath(__file__)
project_folder_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_folder_path)

from config import PROJECT_ROOT

def convert_file(input_path, output_path):
    # 初始化转换器（繁体到简体）
    cc = OpenCC('t2s')
    
    # 设置缓冲区大小（处理大文件时优化内存）
    buffer_size = 100000  # 约100KB
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        buffer = []
        for line in tqdm(f_in, desc="processing: "):
            # 转换当前行
            simplified = cc.convert(line)
            buffer.append(simplified)
            
            # 达到缓冲区大小时写入
            if len(buffer) >= buffer_size:
                f_out.writelines(buffer)
                buffer = []
        
        # 写入剩余内容
        if buffer:
            f_out.writelines(buffer)

input_path = os.path.join(PROJECT_ROOT, '/data/tokenizer_data/wiki.txt')
output_path = os.path.join(PROJECT_ROOT, '/data/tokenizer_data/wiki_s.txt')

# 语料转换为中文
convert_file(input_path, output_path)
