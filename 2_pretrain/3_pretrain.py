import os
import sys
import platform
import time
from dataclasses import dataclass, field
from typing import Optional

current_file_path = os.path.abspath(__file__)
project_folder_path = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_folder_path)

from config import PROJECT_ROOT

import numpy as np
import pandas as pd
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerControl, TrainerState

from datasets import Dataset, load_dataset
from model.qwen.configuration_qwen import QWenConfig
from model.qwen.modeling_qwen import QWenLMHeadModel
# from model.qwen.tokenization_qwen import QWenTokenizer

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

attn_implementation = "flash_attention_2"
try:
    from flash_attn import flash_attn_func
except Exception as e:
    attn_implementation = "eager"


# 筛选18个天工文件和百度百科作为训练集
DATA_PATH = PROJECT_ROOT + '/data/pretrain/processed_data'
TRAIN_FILES = [DATA_PATH + '/' + train_parquet for train_parquet in os.listdir(DATA_PATH) if train_parquet.startswith('2020-40') or train_parquet.startswith('baike')] 

# wifi filter作为验证集
EVAL_FILE = PROJECT_ROOT + "/data/pretrain/processed_data/wiki_fi.parquet"

######################## 预准备部分 ########################
# 训练数据集setting
@dataclass
class PretrainArguments:
    tokenizer_dir: str = PROJECT_ROOT + "/model_save/hf_tokenizer/"
    model_save_dir: str = PROJECT_ROOT + "/model_save/pretrain/"
    logs_dir: str = PROJECT_ROOT + "/logs/"
    train_files: list = field(default_factory=lambda: TRAIN_FILES)
    eval_file: str = EVAL_FILE
    max_seq_len: int = 512

    # Windows 使用默认的attention实现，
    attn_implementation: str = (
        "eager" if platform.system() == "Windows" else attn_implementation
    )

# logs路径
logs_path = PROJECT_ROOT + '/logs'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
####################### 1.训练数据集 #######################
pretrain_args = PretrainArguments()
    
# 显式加载Fast版本的tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=PROJECT_ROOT + "/model_save/hf_tokenizer/tokenizer.json",  # 必须指定tokenizer.json
    config_file=PROJECT_ROOT + "/model_save/hf_tokenizer/tokenizer_config.json",
    special_tokens_map_file=PROJECT_ROOT + "/model_save/hf_tokenizer/special_tokens_map.json",
    padding_side="right",  # 对齐Qwen的默认配置
    truncation_side="right",
    use_fast=True,         # 显式启用快速模式
    trust_remote_code=True # 如果Qwen需要自定义代码
)
######################## 2. 加载训练好的tokenizer #######################

# 如果你使用的`add_tokens`方法添加了自己的token，必须要用`len(tokenizer)`获取长度，`tokenizer.vocab_size`统计不包含你添加的字符。
print(f"词汇表大小: {len(tokenizer)}")

# 更新tokenizer setting
tokenizer.model_max_length = 8192
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # 为了方便cuda计算，词表的大小注意一下，如果不是64的整数倍，可以手动向上取整为64的整数倍，也可以是其他 $2^x$ 数值的整数倍，如32、128、256都行。
vocab_size = len(tokenizer)
if vocab_size % 64 != 0:
    vocab_size = (vocab_size // 64 + 1) * 64
print(f"final vocab size: {vocab_size}")

# 如果词表大小小于 65535 用uint16存储，节省磁盘空间，否则用uint32存储
map_dtype = np.uint16 if vocab_size < 65535 else np.uint32

# tokenizer的映射函数
def token_to_id(samples: dict) -> dict:

    batch_txt = samples["text"]
    outputs = tokenizer(
        batch_txt,
        padding=False,
        return_attention_mask=False,
        truncation=True,
        max_length=pretrain_args.max_seq_len
    )
    input_ids = [np.array(item, dtype=map_dtype) for item in outputs["input_ids"]]

    return {"input_ids": input_ids}

# text_sample = {
#     'text':[
#         '判断给定的文章是否符合语法规则。如果不符合，请提供修改建议。\n',
#         '下面是一篇文章的开头: "为了探讨这个主题，本文将提供一系列数据和实例，以证明这一观点。'
#     ]
# }
# res_encode = token_to_id(text_sample)
# print(res_encode)

# 数据集加载函数
def get_maped_dataset(files) -> Dataset:
    dataset = load_dataset(
        path="parquet",
        data_files=files,
        split="train",
        cache_dir=PROJECT_ROOT + '/data/pretrain/cache',
        keep_in_memory=False,
    )
    maped_dataset = dataset.map(
        token_to_id,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count() // 2,
        keep_in_memory=False,
        writer_batch_size=1000  # 新增：控制磁盘写入批次
    )
    return maped_dataset

############## call back相关设置 ################
class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在打印 n 次日志后清除cuda缓存，适合低显存设备，能防止OOM
        """
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        在on_epoch_end时保存一次模型。
        TrainingArguments的 save_strategy 中 epoch 和 steps 不兼容。要实现每隔 save_steps 步保存一次检查点，考虑到磁盘空间大小，最多只保存最近3个检查点。
        """
        # 设置should_save=True并返回即可
        control.should_save = True
        return control


if __name__ == '__main__':

    ######################## 3. 加载数据集 #######################
    train_dataset = get_maped_dataset(pretrain_args.train_files)
    eval_dataset = get_maped_dataset(pretrain_args.eval_file)

    print(train_dataset, eval_dataset)

    #################### 4. 定义data_collator #######################
    
    # `mlm=False`表示要训练CLM模型，`mlm=True`表示要训练MLM模型
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # 如果配置了flash_attention_2，请手动设置set_default_dtype为float16
    #  Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes.
    if pretrain_args.attn_implementation == "flash_attention_2":
        torch.set_default_dtype(torch.bfloat16)
    
    #################### 5. 定义模型 #######################
    model_config = QWenConfig.from_pretrained(PROJECT_ROOT + '/model/qwen')
    # 调整setting
    model_config.vocab_size = 41024
    model_config.num_hidden_layers = 16 # 原始：32 ### 需要等比例放缩
    model_config.hidden_size = 1024 # 原始：2048
    # model_config.intermediate_size = 5504
    model_config.kv_channels = 64 # 原始：128
    model_config.pad_token_id = tokenizer.pad_token_id
    
    # 加载模型
    model = QWenLMHeadModel(model_config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"QWen size: {model_size / 1000**2:.1f}M parameters")
    print(f"模型参数量: {model_size / 1e9:.1f}B")
    
    # 训练器callback模块
    my_trainer_callback = MyTrainerCallback()
    
    #################### 6. 定义训练参数 #######################
    args = TrainingArguments(
        output_dir=pretrain_args.model_save_dir,
        per_device_train_batch_size=4, # 原始：24
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=10,
        num_train_epochs=1,
        weight_decay=0.1,
        ddp_find_unused_parameters=False,
        warmup_steps=0,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=50,
        save_strategy="steps",
        save_total_limit=4,
        report_to="tensorboard",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        log_level="info",
        logging_first_step=True,
        # group_by_length=True,
        # deepspeed='./ds_config_one_gpu.json',
    )
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[my_trainer_callback],
    )
    
    #################### 7. 开始训练 #######################
    # `resume_from_checkpoint=True`参数可以从上次保存的检查点继续训练
    trainer.train(
        # resume_from_checkpoint=True
        )
    
    # 记录训练损失
    loss_log = pd.DataFrame(trainer.state.log_history)
    loss_log.to_csv(PROJECT_ROOT + f"/logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
    
    #################### 8. 训练后验证 #######################
    eval_results = trainer.evaluate()
    print(f"Perplexity: {np.exp(eval_results['eval_loss']):.2f}")
    
    # 保存模型
    trainer.save_model(pretrain_args.model_save_dir)
    
    print('')