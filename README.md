# MyLLM_stepBystep

用于记录个人从0开始实践一个小量级的llm的心路历程（x

流程会在日后实践中逐步完善，边做边学习

· 考虑实践0.5B，1B量级的模型，准备使用的基模型为Qwen2和Llama2

# 一、数据收集

## 1.1 公开数据集

借鉴下目前其他优秀项目收集的优质公开数据集

| pretrain数据集链接 | 描述  |
| --- | --- |
| Wiki中文百科：[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered) | 中文Wikipedia的数据 |
| BaiduBaiKe：[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) 提取码: bwvb | 中文BaiduBaiKe的数据 |
| C4_zh：[百度网盘 part1](https://pan.baidu.com/s/18O2Tj_PPB718K8gnaWrWUQ) 提取码：zv4r；[百度网盘 part2](https://pan.baidu.com/s/11PTgtUfFXvpNkOige9Iw4w) 提取码：sb83；[百度网盘 part3](https://pan.baidu.com/s/1248QfTS8QHPojYW-0fd5jQ) 提取码：l89d | C4是可用的最大语言数据集之一，收集了来自互联网上超过3.65亿个域的超过1560亿个token。C4_zh是其中的一部分 |
| WuDaoCorpora：[智源研究院BAAI：WuDaoCorpora Text文本预训练数据集](https://data.baai.ac.cn/details/WuDaoCorporaText) |     |

| tokenizer数据集链接 | 描述 |
| --- | --- |
| wiki语料[wiki.txt](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles-multistream.xml.bz2) | 中文wiki数据，用于训练tokenizer |

# 二、预训练(Pretrain)

## 2.1 预训练预料处理
1. 预训练预料处理
2. 分词器预料处理

下载后将源文件放入`/data/tokenizer_data`下，提取其中的预料内容,执行`/1_data_process`中的
```bash
python Tokenizer_1_WikiExtractor.py --infn /data/tokenzier_data/zhwiki-latest-pages-articles-multistream.xml.bz2
```

第二步将繁体转换为中文，名称可以对应调整，wiki_s为对应的简体版本
```bash
python Tokenizer_2_convert_t2s.py wiki.txt wiki_s.txt
```

## 2.2 Tokenizer
训练tokenizer，将训练预料准备好并放入指定路径后，执行`2_pretrain`中的
```bash
python 1_train_tokenizer.py
```
训练好后的Tokenizer存储在model_save路径下

### （可能涉及的框架/库：DeepSpeed, Megatron-LM, transformers...）

# 三、后训练(Post-training)

## 1. SFT全量微调

### 相关框架：OpenRLHF, LLaMA-Factory

### 1.1 微调数据

**日常问答SFT数据**：

| SFT语料 | 描述  |
| --- | --- |
| alpaca-zh：[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh) | 源自shibing624的一部分SFT数据。该数据集是参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条。 |
| bell：[bell](https://huggingface.co/datasets/BelleGroup/train_1M_CN) | 源自BelleGroup的一部分SFT数据。包含约100万条由BELLE项目生成的中文指令数据。 |

## 2. SFT-LoRA微调

## 3. RHLF

###     3.1 DPO

###     3.2 PPO

###     3.3 GRPO（*）

# 四、推理与评测

## 1. 推理部署

###     1.1 LMdeploy

###     1.2 vLLM

###     *待补充

## 2. 推理

###     1.1 opencompass

###     *待补充

# 五、其他应用框架

## 1. Langchain

### 2. AutoGen

### 待补充

# 参考项目链接

ChatLM-mini_Chinses: [ChatLM-mini_Chinese](https://github.com/charent/ChatLM-mini-Chinese)

MINI_LLM: [MINI_LLM](https://github.com/jiahe7ay/MINI_LLM)

baby-llama2: [baby-llama2](https://github.com/DLLXW/baby-llama2-chinese)
