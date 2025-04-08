from transformers import AutoTokenizer
import os
import sys
current_file_path = os.path.abspath(__file__)
project_folder_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_folder_path)
from config import PROJECT_ROOT

# 1. 加载训练好的tokenizer（替换为你的tokenizer文件夹路径）
tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, 'model_save\hf_tokenizer'))

# 2. 基础编码/解码测试
text = "这是一个测试文本，This is a test text."  # 测试文本
# print("\n原始文本:", text)

# 编码（文本 -> IDs）
encoded = tokenizer.encode(text, add_special_tokens=True)
print("\n编码结果（带特殊标记）:", encoded)

# 解码（IDs -> 文本）
decoded = tokenizer.decode(encoded, skip_special_tokens=True)
print("\n解码结果（跳过特殊标记）:", decoded)

# 3. 查看分词细节
tokens = tokenizer.tokenize(text)
print("\n分词结果:", tokens)

# 将token转换为ID
ids = tokenizer.convert_tokens_to_ids(tokens)
print("\nTokens转IDs:", ids)

# 4. 批量编码测试
batch_texts = ["第一条测试文本", "Second test text with symbols: !@#$%^"]
batch_encodings = tokenizer(batch_texts, padding=True, truncation=True)

print("\n批量编码结果:")
print("Input IDs:", batch_encodings.input_ids)
print("Attention Mask:", batch_encodings.attention_mask)

# 5. 特殊标记测试
print("\n特殊标记:")
print("CLS token:", tokenizer.cls_token, "ID:", tokenizer.cls_token_id)
print("SEP token:", tokenizer.sep_token, "ID:", tokenizer.sep_token_id)
print("PAD token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)
print("UNK token:", tokenizer.unk_token, "ID:", tokenizer.unk_token_id)

# 6. 测试未知字符（可选）实际上语料已经包含，输出的是正常的日文
unknown_text = "こんにちは"  # 日文测试（假设你的tokenizer未训练日文字符）
unknown_encoded = tokenizer.encode(unknown_text)
print("\n未知字符处理:", unknown_encoded)
print("解码未知字符:", tokenizer.decode(unknown_encoded))