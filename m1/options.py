# 导入必要的库
import os
import time
import argparse
import torch

def get_options(args=None):
    """
    获取并解析命令行参数，用于配置训练和模型参数。

    参数:
    args -- 可选参数列表，默认为None，将从命令行读取参数

    返回:
    opts -- 解析后的参数对象
    """

    # 创建解析器，并设置程序描述
    parser = argparse.ArgumentParser(description="基于注意力机制的模型，利用强化学习解决旅行商问题")

    # 数据相关参数配置
    parser.add_argument('--problem', default='tsp', help="要解决的问题，默认为'tsp'")
    parser.add_argument('--graph_size', type=int, default=19, help="问题图的大小")
    parser.add_argument('--batch_size', type=int, default=512, help='训练期间每批实例的数量')
    parser.add_argument('--epoch_size', type=int, default=128000, help='训练期间每轮的实例数量')
    parser.add_argument('--val_size', type=int, default=1000,help='用于报告验证性能的实例数量')
    parser.add_argument('--val_dataset', type=str, default=None, help='用于验证的数据集文件')

    # 模型参数配置
    parser.add_argument('--model', default='attention', help="模型类型，'attention'（默认）或'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='输入嵌入的维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='编码器/解码器中隐藏层的维度')
    parser.add_argument('--n_encode_layers', type=int, default=2,help='编码器/评论网络中的层数')
    parser.add_argument('--tanh_clipping', type=float, default=10.,help='使用tanh将参数限制在+-此值内。设置为0表示不进行任何限制。')
    parser.add_argument('--normalization', default='batch', help="归一化类型，'batch'（默认）或'instance'")

    # 训练参数配置
    parser.add_argument('--lr_model', type=float, default=1e-4, help="行为网络的学习率")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="评论网络的学习率")
    parser.add_argument('--lr_decay', type=float, default=0.995, help='每轮学习率衰减')
    parser.add_argument('--eval_only', action='store_true', help='仅评估模型')
    parser.add_argument('--n_epochs', type=int, default=2, help='训练的轮数')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,help='梯度裁剪的最大L2范数，默认1.0（0表示禁用裁剪）')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8, help='指数移动平均基线衰减（默认0.8）')
    parser.add_argument('--baseline', default=None,help="基线类型：'rollout'、'critic'或'exponential'。默认无基线。")
    # parser.add_argument('--baseline', default='rollout',help="基线类型：'rollout'、'critic'或'exponential'。默认无基线。")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='更新rollout基线时的t检验显著性')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='基线预热轮数，对于rollout默认为1（预热阶段使用指数），否则为0。只能与rollout基线一起使用。')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="基线评估时使用的批处理大小")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='启用以减少内存使用，通过检查点保存编码器状态')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='如果批次中有至少这么多实例完成，则缩小批次大小以节省内存（默认为None，表示不缩小）')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='训练期间使用的数据分布，取决于具体问题，默认和选项因问题而异。')

    # 日志和输出目录参数配置
    parser.add_argument('--log_step', type=int, default=50, help='每log_step步记录信息')
    parser.add_argument('--log_dir', default='logs', help='写入TensorBoard信息的目录')
    parser.add_argument('--run_name', default='run', help='运行标识名称')
    parser.add_argument('--output_dir', default='outputs', help='写入输出模型的目录')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='开始轮次（影响学习率衰减）')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='每n轮保存一次检查点（默认1），0表示不保存检查点')
    parser.add_argument('--load_path', help='加载模型参数和优化器状态的路径')
    parser.add_argument('--resume', help='从上一个检查点文件继续')
    parser.add_argument('--no_tensorboard', action='store_true', help='禁用TensorBoard文件的日志记录')
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')

    # 解析参数并返回
    opts = parser.parse_args(args)

    # 根据CUDA是否可用和用户设置，决定是否使用CUDA
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    # 为运行命名添加时间戳
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    # 构建模型输出目录路径
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    # 设置基线预热轮数的默认逻辑
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    # 确保配置逻辑的一致性
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "轮次大小必须是批次大小的整数倍！"

    return opts
