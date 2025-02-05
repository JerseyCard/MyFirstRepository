


# 使用更简单的分句方法，避免使用 NLTK
# 如果你一定要使用 NLTK 的分句功能，需要确保正确下载所有必要资源：

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载所需资源
nltk.download('punkt')
nltk.download('punkt_tab')





from datasets import load_dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 准备并标记数据集
# 加载billsum数据集，选择200个样本并随机打乱
billsum = load_dataset("billsum", split="ca_test").shuffle(seed=42).select(range(200))
# 将数据集分割为训练集和测试集
billsum = billsum.train_test_split(test_size=0.2)
# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("t5-small")
prefix = "summarize: "

# 定义预处理函数，将文本转换为模型输入格式
def preprocess_function(examples):
    # 为每个输入文本添加前缀
    inputs = [prefix + doc for doc in examples["text"]]
    # 将输入文本转换为token，并限制最大长度为1024
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # 将目标摘要转换为token，并限制最大长度为128
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    # 将标签添加到模型输入中
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 对整个数据集应用预处理函数
tokenized_billsum = billsum.map(preprocess_function, batched=True)

# 确保 NLTK 数据被正确下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# 设置评估指标
# nltk.download("punkt", quiet=True)

# 加载ROUGE评估指标
metric = evaluate.load("rouge")

# 定义计算评估指标的函数
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # 解码预测结果和标签
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 将预测和标签按句子分割，并用换行符连接
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # 计算ROUGE分数
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

# 加载预训练模型和设置训练参数
# 初始化T5-small模型
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
# 初始化数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 设置训练参数
""" training_args = Seq2SeqTrainingArguments(
    output_dir="./results",          # 输出目录
    evaluation_strategy="epoch",     # 每个epoch后进行评估
    learning_rate=2e-5,             # 学习率
    per_device_train_batch_size=16, # 训练批次大小
    per_device_eval_batch_size=4,   # 评估批次大小
    weight_decay=0.01,              # 权重衰减
    save_total_limit=3,             # 保存最多3个检查点
    num_train_epochs=2,             # 训练轮数
    fp16=True,                      # 使用16位精度
    predict_with_generate=True      # 使用生成式预测
) """

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    bf16=True,                      # 使用 bfloat16 而不是 fp16
    predict_with_generate=True
)

# 初始化训练器
trainer = Seq2SeqTrainer(
    model=model,                          # 模型
    args=training_args,                   # 训练参数
    train_dataset=tokenized_billsum["train"], # 训练数据集
    eval_dataset=tokenized_billsum["test"],   # 测试数据集
    tokenizer=tokenizer,                      # 分词器
    data_collator=data_collator,             # 数据整理器
    compute_metrics=compute_metrics          # 评估指标计算函数
)

# 开始训练
trainer.train()