from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np

# 1. 加载模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 准备数据
train_data = [
    {"text": "剧情老套，充满套路和硬凹的感动。", "label": 0},
    {"text": "味道非常一般，跟评论区说的完全不一样。", "label": 0},
    {"text": "电影非常精彩，演员演技在线！", "label": 1},
    {"text": "菜品新鲜，配送速度快，点赞！", "label": 1},
]
dataset = Dataset.from_list(train_data)

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. 训练配置（关键参数）
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=10,
)

# 5. 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.train()

# 6. 预测
sentences = [
    "剧情老套，充满套路和硬凹的感动。",
    "味道非常一般，跟评论区说的完全不一样。"
]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

probabilities = torch.softmax(logits, dim=1)
predictions = np.argmax(probabilities.numpy(), axis=1)

label_map = {0: "负面", 1: "正面"}
for sentence, pred in zip(sentences, predictions):
    print(f"句子：{sentence}")
    print(f"情感倾向：{label_map[pred]}\n")