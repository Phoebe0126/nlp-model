import torch
import torch.nn as nn

input = torch.randn(4, 3)
target = torch.tensor([0, 1, 1, 2])  # 必须为Long类型，是类别的序号
cross_entropy_loss = nn.CrossEntropyLoss()
loss = cross_entropy_loss(input, target)

# 对于序列标注来说，需要reshape一下
input = torch.randn(2, 4, 3)  # 2为batch_size, 4为seq_length，3为类别数
print(input)
input = input.view(-1, 3)  # 一共8个token
target = torch.tensor([[0, 1, 1, 2], [2, 2, 1, 0]])
target = target.view(-1)
print(input, target)
loss = cross_entropy_loss(input, target)  # reduction='mean'，默认为mean；