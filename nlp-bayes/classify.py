import re
import argparse
import numpy as np
from jieba import cut
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)  # 过滤无效字符
            line = cut(line)  # 使用jieba分词
            line = filter(lambda word: len(word) > 1, line)  # 过滤单字词
            words.extend(line)
    return words


# ========== 命令行参数解析 ==========
parser = argparse.ArgumentParser()
parser.add_argument('--feature-type', choices=['count', 'tfidf'], default='count',
                    help='特征选择方法: count(词频) 或 tfidf')
parser.add_argument('--balance', action='store_true',
                    help='启用SMOTE样本平衡')
args = parser.parse_args()

# ========== 数据准备 ==========
# 加载所有邮件的分词结果
all_words = [get_words(f'邮件_files/{i}.txt') for i in range(151)]

# 构建文本语料库（将分词列表转换为空格分隔的字符串）
corpus = [' '.join(words) for words in all_words]
labels = np.array([1] * 127 + [0] * 24)  # 前127为垃圾邮件，后24为普通邮件

# ========== 特征工程 ==========
# 选择特征提取器
if args.feature_type == 'tfidf':
    vectorizer = TfidfVectorizer(max_features=100)  # 保持100个特征维度
else:
    vectorizer = CountVectorizer(max_features=100)

# 生成特征矩阵
X = vectorizer.fit_transform(corpus).toarray()

# ========== 样本平衡处理 ==========
if args.balance:
    X, labels = SMOTE(random_state=42).fit_resample(X, labels)

# ========== 数据集划分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42)

# ========== 模型训练 ==========
model = MultinomialNB()
model.fit(X_train, y_train)

# ========== 模型评估 ==========
y_pred = model.predict(X_test)
print("\n分类评估报告:")
print(classification_report(y_test, y_pred,target_names=['普通邮件', '垃圾邮件']))


# ========== 预测函数 ==========
def predict(filename):
    """对未知邮件分类"""
    # 处理新文本
    words = ' '.join(get_words(filename))  # 保持与训练数据相同的处理流程
    new_vector = vectorizer.transform([words]).toarray()

    # 预测并返回结果
    result = model.predict(new_vector)
    return '垃圾邮件' if result[0] == 1 else '普通邮件'


# ========== 执行预测 ==========
print('\n测试邮件分类结果:')
print('151.txt分类情况:', predict('邮件_files/151.txt'))
print('152.txt分类情况:', predict('邮件_files/152.txt'))
print('153.txt分类情况:', predict('邮件_files/153.txt'))
print('154.txt分类情况:', predict('邮件_files/154.txt'))
print('155.txt分类情况:', predict('邮件_files/155.txt'))