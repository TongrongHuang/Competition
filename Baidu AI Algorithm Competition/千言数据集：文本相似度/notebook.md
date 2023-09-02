# 千言数据集：文本相似度
竞赛地址：[千言数据集：文本相似度](https://aistudio.baidu.com/competition/detail/45/0/task-definition)  
该比赛使用LCQMC、BQ Corpus和PAWS-X (中文)三个公开的文本相似度计算数据集。

计算LCQMC数据集文本相似度的代码如下：
首先导入必要的库

    import paddle  # 使用paddle代码框架
    import paddle.nn.functional as F
    from paddlenlp.datasets import load_dataset  # 使用paddle的千言数据集
    import paddlenlp  # 使用paddle预训练好的自然语言处理模型

    import time
    import os
    import numpy as np

    import warnings
    warnings.filterwarnings("ignore")  # 过滤警告

加载数据集

    train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])  # 加载lcqmc的训练数据集和验证数据集

查看训练集数据前三条样本

    for idx, example in enumerate(train_ds):
        if idx < 3:
            print(example)

    output:
    {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生', 'label': 1}
    {'query': '我手机丢了，我想换个手机', 'title': '我想买个新手机，求推荐', 'label': 1}
    {'query': '大家觉得她好看吗', 'title': '大家觉得跑男好看吗？', 'label': 0}

加载 ERNIE-Gram 的tokenizer对文本进行切分

    tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained("ernie-gram-zh")

将数据的 query、title 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据,返回 input_ids 和 token_type_ids

    def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    
        query, title = example["query"], example["title"]  # 拿到数据的上下文
        
        encoded_inputs = tokenizer(text=query, text_pair=title, max_seq_len=max_seq_length)  # 将明文数据转成ID数据
        
        input_ids = encoded_inputs["input_ids"]  # 拿到词汇表中对应的ID
        token_type_ids = encoded_inputs["token_type_ids"]  # 拿到token对应句子的ID，为0或1，0表示属于token的第一句，1表示属于token的第二句
        
        if not is_test:
            label = np.array([example["label"]], dtype="int64")
            return input_ids, token_type_ids, label
        # 如果是测试不返回标签
        else:
            return input_ids, token_type_ids

对第一条数据进行数据转换

    input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)

查看input_ids的长度，input_ids，token_type_ids以及对应的label

    print(input_ids)  # 查看input_ids
    output:
    [1, 692, 811, 445, 2001, 497, 5, 654, 21, 692, 811, 614, 356, 314, 5, 291, 21, 2, 329, 445, 2001, 497, 5, 654, 21, 692, 811, 614, 356, 314, 5, 291, 21, 2]

    print(len(input_ids))  # 查看input_ids的长度
    output:
    34  # 注意，input_ids的长度比原输入文本长了3个单位，是因为在query开头插入了[CLS]，末尾插入了[SEP]，title末尾插入了[SEP]

    print(token_type_ids)  # 查看token_type_ids
    output:
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

使用python偏函数（partial）给 convert_example 赋予一些默认参数

    from functools import partial

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=512)

训练数据会返回input_ids、token_type_ids、labels三个字段，所以针对这3个字段需要分别定义 3 个组 batch 操作

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # 在第一个维度填充值为token_id
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # 在第一个维度填充值为token_type_ids
        Stack(dtype="int64")  # 将token_id和token_type_id堆叠起来，堆叠后数据类型为int64
    ): [data for data in fn(samples)]  # 使用列表推导式，将每个输入sample应用fn里的三个函数，将结果存在列表中

定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练

    batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)

基于 train_ds 定义 train_data_loader，会自动对数据进行切分

    train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),  # 对数据集每个样本使用trans_func函数
        batch_sampler=batch_sampler,  # 批量采样
        collate_fn=batchify_fn,  # 将样本整理成合适的格式，以便输入模型中
        return_list=True)  # 返回样本列表

针对训练集，使用单卡进行评估

    batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)
    dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络

    pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')  # 加载预训练模型

    class PointwiseMatching(nn.Layer):
   
        # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
        def __init__(self, pretrained_model, dropout=None):
            super().__init__()
            self.ptm = pretrained_model  # 预训练模型
            self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)  # 缓解模型过拟合

            # 语义匹配任务: 相似、不相似，是一个 2 分类任务
            self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

        def forward(self,
                    input_ids,
                    token_type_ids=None,
                    position_ids=None,
                    attention_mask=None):

            # 此处的 Input_ids 由两条文本的 token ids 拼接而成
            # token_type_ids 表示两段文本的类型编码
            # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
            _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                        attention_mask)

            cls_embedding = self.dropout(cls_embedding)

            # 基于文本对的语义表示向量进行 2 分类任务
            logits = self.classifier(cls_embedding)
            probs = F.softmax(logits)  # 拿到文本匹配的概率

            return probs

    model = PointwiseMatching(pretrained_model)  # 定义 Point-wise 语义匹配网络

定义优化器、损失函数和评价指标

    from paddlenlp.transformers import LinearDecayWithWarmup

    epochs = 3
    num_training_steps = len(train_data_loader) * epochs

    # 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
    lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    # 定义 Optimizer
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=0.0,
        apply_decay_param_fun=lambda x: x in decay_params)

    # 采用交叉熵 损失函数
    criterion = paddle.nn.loss.CrossEntropyLoss()

    # 评估的时候采用准确率指标
    metric = paddle.metric.Accuracy()

因为训练过程中同时要在验证集进行模型评估，所以先定义训练过程中的评估函数

    @paddle.no_grad()  # @paddle.no_grad()是一个装饰器，用于将一个函数或方法标记为不使用梯度计算
    def evaluate(model, criterion, metric, data_loader, phase="dev"):
        model.eval()  # 设置为评估模式，不更新参数
        metric.reset()  # 重置评估结果
        losses = []  # 存储每个批次的损失
        for batch in data_loader:
            input_ids, token_type_ids, labels = batch  # 获取每个批量样本的input_ids、token_type_ids和labels
            probs = model(input_ids=input_ids, token_type_ids=token_type_ids)  # 进行预测
            loss = criterion(probs, labels)  # 计算标签之间的损失
            losses.append(loss.numpy())
            correct = metric.compute(probs, labels)  # 计算预测结果的正确性
            metric.update(correct)  # 更新到度量指标
            accu = metric.accumulate()  # 计算度量指标累计值
        print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                        np.mean(losses), accu))  # 输出损失、准确率
        model.train()  # 调整为训练模式
        metric.reset()  # 重置评估结果

预测函数

    def predict(model, data_loader):
    
        batch_probs = []

        model.eval()  # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉

        with paddle.no_grad():
            for batch_data in data_loader:
                input_ids, token_type_ids = batch_data
                input_ids = paddle.to_tensor(input_ids)
                token_type_ids = paddle.to_tensor(token_type_ids)
                
                # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
                batch_prob = model(
                    input_ids=input_ids, token_type_ids=token_type_ids).numpy()

                batch_probs.append(batch_prob)
            batch_probs = np.concatenate(batch_probs, axis=0)

            return batch_probs

将预测数据转换成与训练数据一样的形状

    # predict 数据没有 label, 因此 convert_exmaple 的 is_test 参数设为 True
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=512,
        is_test=True)

    # 预测数据的组 batch 操作
    # predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]

    # 加载预测数据
    test_ds = load_dataset("lcqmc", splits=["test"])

单卡预测

    batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=32, shuffle=False)

    # 生成预测数据 data_loader
    predict_data_loader =paddle.io.DataLoader(
            dataset=test_ds.map(trans_func),
            batch_sampler=batch_sampler,
            collate_fn=batchify_fn,
            return_list=True)

加载下载好的模型参数

    state_dict = paddle.load("your_model_parameter_path")  # 将`your_model_parameter_path`修改为你下载好的模型参数的路径

    model.set_dict(state_dict)  # 将模型参数部署到模型网络中

查看一个批次的样本数据

    for idx, batch in enumerate(predict_data_loader):
        if idx < 1:
            print(batch)

    output:
    [Tensor(shape=[32, 38], dtype=int32, place=Place(gpu_pinned), stop_gradient=True,
            [[1   , 1022, 9   , ..., 0   , 0   , 0   ],
                [1   , 514 , 904 , ..., 0   , 0   , 0   ],
                [1   , 47  , 10  , ..., 0   , 0   , 0   ],
                ...,
                [1   , 733 , 404 , ..., 0   , 0   , 0   ],
                [1   , 134 , 170 , ..., 0   , 0   , 0   ],
                [1   , 379 , 3122, ..., 0   , 0   , 0   ]]), Tensor(shape=[32, 38], dtype=int32, place=Place(gpu_pinned), stop_gradient=True,
            [[0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0],
                ...,
                [0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0],
                [0, 0, 0, ..., 0, 0, 0]])]

执行预测函数

    y_probs = predict(model, predict_data_loader)

根据预测概率获得标签（相似或不相似）

    y_preds = np.argmax(y_probs, axis=1)

根据竞赛要求提交格式制作提交文件

    test_ds = load_dataset("lcqmc", splits=["test"])  # 加载测试集

    # 将预测结果写入文件中
    with open("lcqmc.tsv", 'w', encoding="utf-8") as f:
        f.write("index\tprediction\n")    
        for idx, y_pred in enumerate(y_preds):
            f.write("{}\t{}\n".format(idx, y_pred))
            text_pair = test_ds[idx]
            text_pair["label"] = y_pred
            print(text_pair)
