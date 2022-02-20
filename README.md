# my_framework
A Bonfire with Torch(pytorch)

## 想法

框架目的是为了设计,训练以及导出模型

# 设计(Model)

设计 model 非常简单, 可以将模型拆成多个模块, 各个模块根据需求选择或替换即可, 类似于拼积木.

例如常见的 CNN 类模型, model 拆成 backbone(特征提取网络), neck(特征融合/处理), head(输出头) 三部分, 在对应的模块目录下选择即可,
backbone 即可选择 resnet, 也可以选择 vgg

实现一个 model 要比普通的 torch 模型复杂一些, 需要额外实现一个 train_step, val_step.
添加这两个接口的原因是 model 需要关注数据的解析.

训练/验证的脚本(Runner)为了泛用性, 在训练和验证时丢给模型的是包含 input, label 的一个对象.
为了执行forward等操作, 模型需要解出 input.

同时, 由于解出了 label, train_step & val_step 还可以承担损失/指标计算, 更新等.
但是打日志不放在这里, 因为打日志还会用到其他信息, 交给有 step 信息的 Runner 最好,
train_step & val_step 将日志用到的内容输出后 由 Runner 记录到自己的 Runner.train_outout, Runner.train_outout

# 训练(Runner)

训练模型时, 需要 分析环境, 处理输入数据, 保存中间结果(记录以及checkpoint)


# 导出

方案: torch -> ONNX -> TensorRT,

为了便于使用以及减少开发, 构建模型是只考虑可以转换的算子, 但会限制模型的开发