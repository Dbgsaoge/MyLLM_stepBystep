import ujson
from unicodedata import normalize
# import pandas as pd 
# import numpy as np
# from rich import progress
# from fastparquet import ParquetFile, write
import glob
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys
from tqdm import tqdm
import json
current_file_path = os.path.abspath(__file__)
project_folder_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_folder_path)

from config import PROJECT_ROOT

# punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：\n")
# en_punctuation = ",().!;:"
# zh_punctuation = "，（）。！；："

def split_txt_cropus_to_chunk_data(
    texts: list, batch_size: int = 512**2, max_len: int = 512, window_size: int = 2) -> list:

    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = "".join(buffer)

            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for i in range(0, len(buffer_txt), max_len - window_size):
                chunk_data.append("".join(buffer_txt[i : i + max_len]))
            buffer, buffer_len = [], 0

    return chunk_data

def process_none(s: str) -> str:
    if s:
        return s
    return ""

#———————————————————————————————————————— 数据处理 ————————————————————————————————————————#

# 处理wiki_filtered数据
def gen_wiki_filter(origin_file, output_file=PROJECT_ROOT + '/data/pretrain/processed_data/wiki_fi.parquet'):
    lines = []
    with open(origin_file, "r", encoding="utf-8") as f:
        items = ujson.load(f)
        for item in items:
            lines.append(item["completion"] + "<|im_end|>")
    chunk_data = split_txt_cropus_to_chunk_data(lines)
    tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=50000,
        data_page_size=50000,
    )

# 处理百度百科数据
def gen_baike(origin_file):
    baike_items = []
    eos_token = "<|im_end|>"
    max_len = 512
    batch_size, batch_cnt = 2000000, 0
    with open(origin_file, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break

            item = ujson.loads(line)
            cur_txt, cur_len = [], 0

            if not item["title"]:
                continue

            temp_txt = f"{item['title']}：{process_none(item['summary'])}"

            cur_len += len(temp_txt)
            cur_txt.append(temp_txt)

            for section in item["sections"]:

                # 太长的截断不要了
                if cur_len > max_len:
                    break

                title = f"{section['title']}：" if section["title"] else ""
                temp_txt = f"{title}{process_none(section['content'])}"

                cur_len += len(temp_txt)
                cur_txt.append(temp_txt)
            temp_txt = normalize("NFKC", "".join(cur_txt))

            if len(temp_txt) > max_len:
                # 从 max_len 开始找第一个句号，叹号
                n, i = len(temp_txt), max_len
                while i < n and temp_txt[i] not in ("。", "！"):
                    i += 1
                temp_txt = "".join(temp_txt[0 : i + 1])

                # 添加 eos token
            temp_txt = f"{temp_txt}{eos_token}"

            baike_items.append(temp_txt)

            if len(baike_items) % batch_size == 0:

                chunk_data = split_txt_cropus_to_chunk_data(baike_items)
                tb = pa.Table.from_arrays([chunk_data], names=["text"])

                file_name = PROJECT_ROOT + f'/data/pretrain/processed_data/baike_chunk_512_5.6M_{batch_cnt}.parquet'
                pq.write_table(
                    table=tb,
                    where=file_name,
                    row_group_size=50000,
                )

                print(f"save to {file_name}")

                batch_cnt += 1
                baike_items = []
                
# 处理天工数据集
def gen_sky(input_folder, output_folder=PROJECT_ROOT + '/data/pretrain/processed_data'):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):  # 修改为处理JSON Lines文件
            origin_file = os.path.join(input_folder, filename)
            output_file = os.path.join(
                output_folder, filename.replace(".jsonl", ".parquet")
            )
            print(f"Processing {origin_file}...")

            lines = []
            with open(origin_file, "r", encoding="utf-8") as f:
                for line in f:
                    item = ujson.loads(line)
                    lines.append(item["text"] + "<|im_end|>")  # 确保每行都是一个有效的JSON对象

            if lines:  # 确保文件中有内容
                chunk_data = split_txt_cropus_to_chunk_data(lines)
                tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
                pq.write_table(
                    table=tb,
                    where=output_file,
                    row_group_size=50000,
                    data_page_size=50000,
                )
                print(f"Processed {origin_file} to {output_file}")
            else:
                print(f"No content in {origin_file}. Skipping.")
                
# def process_c4_parquet(
#     output_file=PROJECT_ROOT + "/data/pretrain/processed_data/c4_zh.parquet",
#     batch_size=512**2,  # 262144 characters
#     max_len=512,
#     window_size=2
# ):
#     """处理C4数据集并生成Parquet文件（包含分块逻辑）"""
    
#     # 获取文件列表
#     c4_zh_paths = sorted(glob.glob(PROJECT_ROOT + '/data/pretrain/raw_data/c4-zh/*'))
#     print(f"Found {len(c4_zh_paths)} files")

#     # 初始化缓冲区和计数器
#     buffer, buffer_len = [], 0
#     total_chunks = 0

#     # 创建Parquet写入器
#     schema = pa.schema([pa.field('text', pa.string())])
#     writer = pq.ParquetWriter(output_file, schema=schema, flavor='spark')

#     # 处理文件
#     for file_path in tqdm(c4_zh_paths, desc="Processing"):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     # 解析并预处理文本
#                     text = json.loads(line)['text']
#                     processed_text = text + "<|endoftext|>"
                    
#                     # 更新缓冲区
#                     buffer.append(processed_text)
#                     buffer_len += len(processed_text)

#                     # 批量处理条件
#                     if buffer_len >= batch_size:
#                         chunks = split_txt_cropus_to_chunk_data(buffer, max_len, window_size)
#                         if chunks:
#                             table = pa.Table.from_arrays([pa.array(chunks)], names=["text"])
#                             writer.write_table(table)
#                             total_chunks += len(chunks)
#                         buffer, buffer_len = [], 0  # 重置缓冲区

#                 except Exception as e:
#                     print(f"Error processing line: {e}")
#                     continue

#     # 处理剩余数据
#     if buffer:
#         chunks = split_txt_cropus_to_chunk_data(buffer, max_len, window_size)
#         if chunks:
#             table = pa.Table.from_arrays([pa.array(chunks)], names=["text"])
#             writer.write_table(table)
#             total_chunks += len(chunks)

#     # 关闭写入器
#     writer.close()
#     print(f"Generated {total_chunks} chunks | Saved to {output_file}")
        
if __name__ == '__main__':
    # 处理数据
    gen_wiki_filter(PROJECT_ROOT + '/data/pretrain/raw_data/wikipedia-cn-20230720-filtered.json')
    gen_baike(PROJECT_ROOT + '/data/pretrain/raw_data/563w_baidubaike.json')
    # process_c4_parquet()
    gen_sky(PROJECT_ROOT + '/data/pretrain/raw_data/SkyPile-150B')