# my_framework
A Bonfire with Torch(pytorch)

## 设计想法

框架目的是为了设计,训练以及导出模型

# 设计

设计一个 model 时, 首先需要关注 model 获取输入后如何获取输出, model 可以从各个模块获取所需的对象构建

例如常见的 CNN 类模型, model 选取 backbone(特征提取网络), neck(特征融合/处理), head(输出头) 然后组合三者

在设计 head 时, 还需要进行损失, 指标计算, 虽然和模型无关, 但是放到此处是较为适合的

# 训练

训练模型时, 需要 分析环境, 处理输入数据, 保存结果

# 导出

转 ONNX, 然后转 TensorRT