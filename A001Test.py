# not complete

import evaluate

# 加载评估指标
exact_match = evaluate.load('exact_match')
rouge = evaluate.load('rouge')

# 示例预测和参考文本
predictions = ["这是一个测试句子"]
references = ["这是一个测试句子"]

# 计算 exact match 分数
em_results = exact_match.compute(predictions=predictions, references=references)
print("Exact Match 分数:", em_results)

# 计算 ROUGE 分数
rouge_results = rouge.compute(predictions=predictions, references=references)
print("ROUGE 分数:", rouge_results)