# 导入必要的库
import os
import time
import argparse
import torch


def get_options(args=None):
    """
    获取并解析命令行参数，用于配置训练和模型参数。
    :param args: 可选参数列表，如果不提供，则使用命令行参数。
    :return: opts, 解析后的参数配置。
    """

    # 创建解析器，并设置程序描述
    parser = argparse.ArgumentParser( description="使用强化学习解决旅行商问题的注意力模型")

    # 配置与问题相关的参数
    # 数据
    parser.add_argument('--problem', default='tsp', help="要解决的问题，默认为 'tsp'")
    parser.add_argument('--graph_size', type=int, default=19, help="问题图的大小")
    parser.add_argument('--batch_size', type=int, default=512, help='训练期间每批实例的数量')
    parser.add_argument('--epoch_size', type=int, default=128000, help='训练期间每个周期的实例数量')
    parser.add_argument('--val_size', type=int, default=1000,help='用于报告验证性能的实例数量')
    parser.add_argument('--val_dataset', type=str, default=None, help='用于验证的数据集文件')

    # 配置与模型相关的参数
    # 模型
    parser.add_argument('--model', default='attention', help="模型，'attention'（默认）或 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='输入嵌入的维度')
    parser.add_argument('--hidden_dim', type=int, default=128, help='编码器/解码器中隐藏层的维度')
    parser.add_argument('--n_encode_layers', type=int, default=2, help='编码器/评论网络中的层数')
    parser.add_argument('--tanh_clipping', type=float, default=10., help='使用tanh将参数限制在+-此值范围内。设为0不执行任何限制。')
    parser.add_argument('--normalization', default='batch', help="归一化类型，'batch'（默认）或 'instance'")

    # 配置与训练相关的参数
    # 训练
    parser.add_argument('--lr_model', type=float, default=1e-4, help="设置演员网络的学习率")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="设置评论网络的学习率")
    parser.add_argument('--lr_decay', type=float, default=0.995, help='每周期的学习率衰减')
    parser.add_argument('--eval_only', action='store_true', help='仅评估模型')
    parser.add_argument('--n_epochs', type=int, default=1, help='训练周期数')
    parser.add_argument('--seed', type=int, default=1234, help='使用的随机种子')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,help='梯度裁剪的最大L2范数，默认1.0（0禁用裁剪）')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8, help='指数移动平均基线衰减（默认0.8）')
    parser.add_argument('--baseline', default=None, help="要使用的基线：'rollout'，'critic' 或 'exponential'。默认情况下不使用基线。")
    parser.add_argument('--bl_alpha', type=float, default=0.05, help='t检验中更新rollout基线的显著性')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None, help='用于预热基线的周期数，默认值取决于问题。')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help="评估期间使用的批处理大小")
    parser.add_argument('--checkpoint_encoder', action='store_true', help='通过检查点编码器减少内存使用')
    parser.add_argument('--shrink_size', type=int, default=None, help='如果批量中有至少这么多实例完成则缩小批量大小以节省内存')
    parser.add_argument('--data_distribution', type=str, default=None, help='训练期间使用的数据分布，默认值和选项取决于问题。')

    # 配置与日志和输出相关的参数
    # 其他
    parser.add_argument('--log_step', type=int, default=50, help='每 log_step 步记录信息')
    parser.add_argument('--log_dir', default='logs', help='写入TensorBoard信息的目录')
    parser.add_argument('--run_name', default='run', help='标识运行的名称')
    parser.add_argument('--output_dir', default='outputs', help='写入输出模型的目录')
    parser.add_argument('--epoch_start', type=int, default=0, help='从第 epoch# 开始（对于学习率衰减相关）')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='每 n 周期保存检查点（默认1），0表示不保存检查点')
    parser.add_argument('--load_path', help='加载模型参数和优化器状态的路径')
    parser.add_argument('--resume', help='从上一个检查点文件继续')
    parser.add_argument('--no_tensorboard', action='store_true', help='禁用记录TensorBoard文件')
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')

    # 解析参数
    opts = parser.parse_args(args)

    # 根据CUDA是否可用和用户设置，确定是否使用CUDA
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    # 为运行命名添加时间戳
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    # 构建模型输出目录路径
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    # 设置baseline的预热周期，默认值逻辑
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    # 确保配置一致性，如预热周期只在rollout基线中使用
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    # 确保epoch_size是batch_size的整数倍
    assert opts.epoch_size % opts.batch_size == 0, "每个周期的实例数量必须是每批实例数量的整数倍！"

    # 返回解析后的选项
    return opts
