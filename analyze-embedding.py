import torch
import torch.nn.functional as F

# 加载向量文件
data = torch.load("model_checkpoints/debug_toy/val_embeddings.pt")
resume_embeds = data["resume_embeds"]
job_embeds = data["job_embeds"]

# 计算余弦相似度矩阵
sim_matrix = torch.matmul(resume_embeds, job_embeds.T)

# 计算 top-k 推荐
k = 3
topk = torch.topk(sim_matrix, k=k, dim=1).indices  # 每个简历对应 top-k job 索引

# 评估 Top-1 / Top-3 命中率
labels = torch.arange(sim_matrix.size(0))
top1_correct = (topk[:, 0] == labels).sum().item()
top3_correct = sum([labels[i] in topk[i] for i in range(len(labels))])

acc1 = top1_correct / len(labels)
acc3 = top3_correct / len(labels)

print(f"Top-1 Accuracy: {acc1:.4f}")
print(f"Top-3 Accuracy: {acc3:.4f}")

# 示例输出前几条推荐结果
for i in range(3):
    print(f"Resume {i} → recommended job indices {topk[i].tolist()}")
