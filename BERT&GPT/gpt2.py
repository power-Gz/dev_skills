from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re  # 新增：用于后处理文本

# 1. 加载模型与分词器（强制关闭快速分词）
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

# 2. 创建生成管道（显式指定 CPU）
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # 强制使用 CPU
)

# 3. 定义输入并生成文本
prompt = "在一个没有网络的世界里"
generated_texts = generator(
    prompt,
    max_length=50,
    num_return_sequences=1,
    temperature=0.6,        # 进一步降低随机性
    top_p=0.9,
    top_k=30,              # 限制候选词数量
    repetition_penalty=1.6, # 更强力抑制重复
    do_sample=True,
    truncation=True,
    pad_token_id=tokenizer.eos_token_id
)

# 4. 后处理：去除空格合并文本
raw_text = generated_texts[0]["generated_text"]
cleaned_text = re.sub(r'\s+', '', raw_text)  # 删除所有空格

# 5. 输出结果
print("优化后生成结果：")
print(cleaned_text)