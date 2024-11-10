# import torch
# import torch.nn.functional as F
# import numpy as np
# from torch import nn
# import math
# from typing import NamedTuple
# import warnings
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# def _mask_long2byte(mask, n=None):
#     """
#     将长整型掩码转换为按字节表示的布尔掩码。
#
#     参数:
#     - mask: 长整型的掩码，假设每个长整型表示8个布尔值。
#     - n: 可选参数，指定输出掩码的长度。如果未提供，则默认为输入掩码长度的8倍。
#
#     返回:
#     - 返回一个布尔掩码，其中每个元素表示原始掩码中对应位是否为1。
#     """
#     if n is None:
#         n = 8 * mask.size(-1)
#     return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.bool).view(*mask.size()[:-1], -1)[..., :n]
#
# def _mask_byte2bool(mask, n=None):
#     """
#     将按字节表示的掩码转换为布尔掩码。
#
#     参数:
#     - mask: 按字节表示的掩码，其中每个字节的8位表示8个布尔值。
#     - n: 可选参数，指定输出掩码的长度。如果未提供，则默认为输入掩码长度的8倍。
#
#     返回:
#     - 返回一个布尔掩码，其中每个元素表示原始掩码中对应位是否为1。
#     """
#     if n is None:
#         n = 8 * mask.size(-1)
#     return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0
#
# def mask_long2bool(mask, n=None):
#     """
#     将长整型掩码转换为布尔掩码。
#
#     参数:
#     - mask: 长整型的掩码，假设每个长整型表示8个布尔值。
#     - n: 可选参数，指定输出掩码的长度。如果未提供，则默认为输入掩码长度的8倍。
#
#     返回:
#     - 返回一个布尔掩码，其中每个元素表示原始掩码中对应位是否为1。
#     """
#     assert mask.dtype == torch.int64
#     return _mask_byte2bool(_mask_long2byte(mask), n=n)
#
#
#
# def mask_long_scatter(mask, values, check_unset=True):
#     """
#     在给定的mask张量的最后一个维度中，根据values张量的值设置位。
#     该函数支持任意批量维度。如果values包含-1，则不设置任何位。
#     注意：该函数不支持一次设置多个值（与普通的scatter函数不同）。
#
#     参数:
#     - mask: 一个张量，其大小的最后一个维度与values张量的大小匹配。
#             用于指示位设置的位置。
#     - values: 一个张量，包含要在mask中设置的值。其大小必须与mask.size()[:-1]匹配。
#     - check_unset: 一个布尔值，指示是否检查要设置的位是否已经设置。
#                    如果为True且位已设置，则会引发断言错误。
#
#     返回:
#     - 返回一个新的张量，其中根据values在mask中设置了相应的位。
#     """
#     # 确保mask和values的大小除最后一个维度外都一致
#     assert mask.size()[:-1] == values.size()
#     # 创建一个从0到mask最后一个维度大小的范围张量
#     rng = torch.arange(mask.size(-1), out=mask.new())
#     # 将values扩展到与mask相同的维度，以便进行广播操作
#     values_ = values[..., None]
#     # 确定在mask的哪个值上设置位
#     where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
#     # 可选：检查位是否已经设置
#     assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
#     # 通过将1左移适当的位置来设置位
#     return mask | (where.long() << (values_ % 64))
#
#
#
# class SkipConnection(nn.Module):
#
#     def __init__(self, module):
#         # 初始化函数
#         super(SkipConnection, self).__init__()
#         # 将传入的模块存储为实例变量，以便在skip connection中使用
#         self.module = module
#
#     def forward(self, input):
#         """
#         前向传播函数，实现输入的前向计算。
#
#         参数:
#         input (Tensor): 输入张量，作为前向传播的输入。
#
#         返回:
#         Tensor: 输出张量，为输入张量与某个内部模块处理后的结果相加。
#         """
#         # 将输入张量与通过self.module模块处理后的结果相加
#         return input + self.module(input)
#
#
# class MultiHeadAttention(nn.Module):
#     class MultiHeadAttention(nn.Module):
#         """
#         多头注意力机制类，继承自nn.Module。
#
#         参数:
#         - n_heads: int，注意力头的数量。
#         - input_dim: int，输入的维度。
#         - embed_dim: int，嵌入的维度，默认为None。如果提供，则val_dim将基于此值和n_heads计算。
#         - val_dim: int，值维度，默认为None。如果未提供，则基于embed_dim和n_heads计算。
#         - key_dim: int，键维度，默认为None。如果未提供，则与val_dim相同。
#
#         该类实现了多头注意力机制，允许更细粒度的注意力分配。
#         """
#
#         def __init__(
#                 self,
#                 n_heads,
#                 input_dim,
#                 embed_dim=None,
#                 val_dim=None,
#                 key_dim=None
#         ):
#             # 初始化父类nn.Module
#             super(MultiHeadAttention, self).__init__()
#
#             # 如果val_dim未提供，则根据embed_dim和n_heads计算val_dim
#             if val_dim is None:
#                 assert embed_dim is not None, "Provide either embed_dim or val_dim"
#                 val_dim = embed_dim // n_heads
#             # 如果key_dim未提供，则设置为与val_dim相同
#             if key_dim is None:
#                 key_dim = val_dim
#
#             # 初始化类属性
#             self.n_heads = n_heads
#             self.input_dim = input_dim
#             self.embed_dim = embed_dim
#             self.val_dim = val_dim
#             self.key_dim = key_dim
#
#             # 注意力缩放因子，用于缩放注意力分数（见“Attention is all you need”论文）
#             self.norm_factor = 1 / math.sqrt(key_dim)
#
#             # 初始化可学习的参数，用于计算query、key和value
#             self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
#             self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
#             self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
#
#             # 如果embed_dim不为None，则初始化用于输出转换的参数
#             if embed_dim is not None:
#                 self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
#
#             # 初始化参数
#             self.init_parameters()
#
#     def init_parameters(self):
#         """
#         初始化模型参数。
#
#         使用均匀分布对模型的所有参数进行初始化。初始化的范围基于每个参数的维度大小，
#         以确保参数在合理的范围内，有助于模型的稳定训练和收敛。
#
#         遍历模型的所有参数，并计算参数的标准化值（stdv），然后使用这个标准化值作为参数初始化的范围。
#         """
#         for param in self.parameters():
#             stdv = 1. / math.sqrt(param.size(-1))
#             param.data.uniform_(-stdv, stdv)
#
#     def forward(self, q, h=None, mask=None):
#         """
#         前向传播函数，实现多头自注意力机制。
#
#         :param q: 查询张量 (batch_size, n_query, input_dim)
#         :param h: 数据张量 (batch_size, graph_size, input_dim), 若为None，则计算自注意力。
#         :param mask: 掩码张量 (batch_size, n_query, graph_size) 或可转换为该形状，如果n_query == 1，则可以是二维。
#                      掩码中，如果关注不可用，则应包含1（即掩码为负邻接矩阵）。
#         :return: 输出张量 (batch_size, n_query, embed_dim)
#         """
#         # 如果h未提供，则使用q进行自注意力计算
#         if h is None:
#             h = q  # 计算自注意力
#
#         # h的形状应为 (batch_size, graph_size, input_dim)
#         batch_size, graph_size, input_dim = h.size()
#         n_query = q.size(1)
#         # 确保q的batch_size和input_dim与h一致
#         assert q.size(0) == batch_size
#         assert q.size(2) == input_dim
#         assert input_dim == self.input_dim, "Wrong embedding dimension of input"
#
#         # 将h和qreshape成(batch_size * graph_size, input_dim)和(batch_size * n_query, input_dim)
#         hflat = h.contiguous().view(-1, input_dim)
#         qflat = q.contiguous().view(-1, input_dim)
#
#         # 定义keys和values的形状为(n_heads, batch_size, graph_size, key/val_size)
#         shp = (self.n_heads, batch_size, graph_size, -1)
#         shp_q = (self.n_heads, batch_size, n_query, -1)
#
#         # 计算查询张量 (n_heads, n_query, graph_size, key/val_size)
#         Q = torch.matmul(qflat, self.W_query).view(shp_q)
#         # 计算键和值张量 (n_heads, batch_size, graph_size, key/val_size)
#         K = torch.matmul(hflat, self.W_key).view(shp)
#         V = torch.matmul(hflat, self.W_val).view(shp)
#
#         # 计算兼容性 (n_heads, batch_size, n_query, graph_size)
#         compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
#
#         # 如果提供掩码，则应用以防止关注
#         if mask is not None:
#             mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
#             compatibility[mask] = -np.inf
#
#         # 计算注意力分布
#         attn = F.softmax(compatibility, dim=-1)
#
#         # 如果有节点没有邻居，则softmax可能返回nan，我们将其修复为0
#         if mask is not None:
#             attnc = attn.clone()
#             attnc[mask] = 0
#             attn = attnc
#
#         # 使用注意力加权值
#         heads = torch.matmul(attn, V)
#
#         # 将多头注意力输出reshape并乘以输出权重矩阵
#         out = torch.mm(
#             heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
#             self.W_out.view(-1, self.embed_dim)
#         ).view(batch_size, n_query, self.embed_dim)
#
#         return out
#
#
# class Normalization(nn.Module):
#
#     # 定义初始化函数，用于初始化Normalization类的实例
#     # 参数embed_dim指定嵌入的维度大小
#     # 参数normalization指定使用的归一化类型，可以是'batch'（批归一化）或'instance'（实例归一化），默认为'batch'
#     def __init__(self, embed_dim, normalization='batch'):
#         super(Normalization, self).__init__()  # 调用父类的初始化方法
#
#         # 定义一个字典，映射不同类型的归一化方法到对应的PyTorch类
#         normalizer_class = {
#             'batch': nn.BatchNorm1d,
#             'instance': nn.InstanceNorm1d
#         }.get(normalization, None)  # 根据传入的normalization类型获取对应的归一化类，如果类型不匹配，则返回None
#
#         # 初始化归一化器，使用获取的归一化类，设置嵌入维度大小，并启用仿射变换（affine=True）
#         self.normalizer = normalizer_class(embed_dim, affine=True)
#
#         # 默认情况下，归一化初始化仿射参数时，偏置为0，权重为0到1之间的均匀分布，这些值通常太大
#         # 因此，注释掉下面的参数初始化函数调用，以避免使用这些默认的、可能不合适的初始化值
#         # self.init_parameters()
#
#     def init_parameters(self):
#         """
#         初始化模型参数。
#
#         使用均匀分布对模型的所有参数进行初始化。初始化的范围基于每个参数的维度大小，
#         以确保参数在合理的范围内，有助于模型的稳定训练和收敛。
#         """
#         # 遍历模型的所有参数
#         for name, param in self.named_parameters():
#             # 计算标准差，用于确定参数初始化的范围
#             stdv = 1. / math.sqrt(param.size(-1))
#             # 使用均匀分布初始化参数，范围为[-stdv, stdv]
#             param.data.uniform_(-stdv, stdv)
#
#
#     def forward(self, input):
#         """
#         根据正常化层的类型对输入进行不同方式的正常化处理。
#
#         该方法首先检查正常化层的类型，然后对输入进行相应的形状变换，
#         以便正确地应用批量归一化（BatchNorm1d）或实例归一化（InstanceNorm1d）。
#         如果正常化层的类型未知或未定义，将直接返回输入。
#
#         参数:
#         input (Tensor): 模型的输入张量。
#
#         返回:
#         Tensor: 经过正常化处理后的张量。
#         """
#         # 检查正常化层是否为批量归一化层
#         if isinstance(self.normalizer, nn.BatchNorm1d):
#             # 如果是，将输入重塑为批量归一化所需的形状，进行归一化后恢复原始形状
#             return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
#         # 检查正常化层是否为实例归一化层
#         elif isinstance(self.normalizer, nn.InstanceNorm1d):
#             # 如果是，将输入重新排列为实例归一化所需的形状，进行归一化后恢复原始形状
#             return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
#         else:
#             # 如果正常化层的类型未知，断言其应为None，否则抛出异常
#             assert self.normalizer is None, "Unknown normalizer type"
#             # 如果没有正常化层，直接返回输入
#             return input
#
#
#
# class MultiHeadAttentionLayer(nn.Sequential):
#
#     def __init__(self, n_heads, embed_dim, feed_forward_hidden=512, normalization='batch'):
#         """
#         初始化多头注意力层。
#
#         参数：
#         - n_heads: 注意力头的数量。
#         - embed_dim: 词嵌入的维度。
#         - feed_forward_hidden: 前馈神经网络的隐藏层维度，默认为512。
#         - normalization: 标准化方法，可选值为'batch'或'layer'。
#         """
#         # 调用父类构造方法初始化SkipConnection和Normalization等组件
#         # 初始化多头注意力机制，并将其包装在SkipConnection中
#         super(MultiHeadAttentionLayer, self).__init__(
#             SkipConnection(
#                 MultiHeadAttention(
#                 )
#             ),
#             # 初始化标准化层
#             Normalization(embed_dim, normalization),
#             # 初始化前馈神经网络，并将其包装在SkipConnection中
#             SkipConnection(
#                 nn.Sequential(
#                     nn.Linear(embed_dim, feed_forward_hidden),
#                     nn.ReLU(),
#                     nn.Linear(feed_forward_hidden, embed_dim)
#                 ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
#             ),
#             # 再次初始化标准化层
#             Normalization(embed_dim, normalization)
#         )
#
#
# class GraphAttentionEncoder(nn.Module):
#     def __init__(
#             self,
#             n_heads,  # 注意力头数
#             embed_dim,  # 嵌入维度
#             n_layers,  # 层数
#             node_dim=None,  # 节点维度，用于输入嵌入
#             normalization='batch',  # 正则化方法，默认为批处理正则化
#             feed_forward_hidden=512  # 前向传播隐藏层的维度，默认为512
#     ):
#         super(GraphAttentionEncoder, self).__init__()
#
#         # 用于将输入映射到嵌入空间
#         self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
#
#         # 构建多层图注意力层
#         self.layers = nn.Sequential(*(
#             MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
#             for _ in range(n_layers)
#         ))
#
#     def forward(self, x, mask=None):
#         """
#         前向传播方法，用于计算图中节点的嵌入向量。
#
#         参数:
#         - x: 输入张量，表示图中节点的特征。
#         - mask: 掩码张量，目前尚未支持。
#
#         返回值:
#         - h: 节点的嵌入向量，形状为(batch_size, graph_size, embed_dim)。
#         - h.mean(dim=1): 图的嵌入向量，通过对节点嵌入向量求平均得到，形状为(batch_size, embed_dim)。
#         """
#         # 确认mask目前不支持
#         assert mask is None, " mask not yet supported!"
#
#         # 批量乘法以获取节点的初始嵌入
#         if self.init_embed is not None:
#             # 如果init_embed不为空，使用它对输入x进行处理
#             h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
#         else:
#             # 如果init_embed为空，直接使用输入x
#             h = x
#
#         # 通过层堆叠进行节点嵌入计算
#         h = self.layers(h)
#
#         # 返回节点嵌入向量及其平均值（图的嵌入）
#         return h, h.mean(dim=1)
#
# def get_costs(dataset, pi, state, mat):
#     """
#     计算给定数据集和状态下的成本。
#
#     参数:
#     - dataset: 输入的数据集，用于从中提取最大值的索引。
#     - pi: 一个张量，其大小决定了返回的depots张量的尺寸。
#     - state: 当前状态的对象，包含之前的操作和长度信息。
#     - mat: 一个矩阵对象，提供变异操作和成本计算的方法。
#
#     返回:
#     - 成本张量，包括基于当前状态和操作的计算结果。
#     - None，作为统一返回值的一部分，表示没有额外的数据返回。
#     """
#     # 初始化depots张量，大小为pi的行数，每行一个零，表示初始仓库位置。
#     depots = torch.zeros(pi.size(0), 1).long().to(device)
#     # 从数据集中找到最大值的索引，用于后续计算。
#     _, ind = torch.max(dataset, dim=2)
#     # 根据上一个动作和矩阵的特性，计算变异值。
#     bdd = mat.var[state.prev_a.squeeze() * mat.n_c].unsqueeze(1)
#     # 生成一个正态分布的随机数张量，用于模拟变异。
#     bdd = torch.randn(state.prev_a.size(0), device=device) * bdd
#     # 计算额外的成本。
#     add = mat.__getd__(ind, state.prev_a, depots, state.lengths).unsqueeze(1)
#     # 将变异值重复，以便在两个不同的上下文中使用。
#     bdd = bdd.repeat(1, 2)
#     # 在第二个位置上应用第一个计算的成本。
#     bdd[:, 1] = add.squeeze() * 5
#     # 选择每行的最小值，进行成本优化。
#     bdd = torch.min(bdd, dim=1)[0]
#     # 将最小值张量扩展，以便进行后续计算。
#     bdd = bdd[:, None].repeat(1, 2)
#     # 在第二个位置上应用第二个计算的成本。
#     bdd[:, 1] = add.squeeze() * -0.9
#     # 选择每行的最大值，进行成本调整。
#     bdd = torch.max(bdd, dim=1)[0]
#     # 返回最终计算的成本，包括当前长度、添加的成本和变异的影响。
#     return state.lengths.squeeze() + add.squeeze() + bdd.squeeze(), None
#
#
# class StateTSP(NamedTuple):
#     # 定义状态变量loc，表示位置信息，使用torch.Tensor类型存储
#     loc: torch.Tensor
#
#     # 在某些情况下，如同一个实例的多个副本（比如束搜索），为了内存效率，不会多次保存loc和dist张量
#     # 而是使用ids来索引正确的行
#     ids: torch.Tensor  # 保存原始固定数据行的索引
#
#     # 定义状态变量first_a，表示第一步的动作
#     first_a: torch.Tensor
#     # 定义状态变量prev_a，表示上一步的动作
#     prev_a: torch.Tensor
#     # 定义状态变量visited_，记录已经访问过的节点
#     visited_: torch.Tensor
#     # 定义状态变量lengths，存储路径长度
#     lengths: torch.Tensor
#     # 定义状态变量cur_coord，表示当前位置坐标
#     cur_coord: torch.Tensor
#     # 定义状态变量i，记录当前步骤数
#     i: torch.Tensor
#
#     @property
#     def visited(self):
#         """
#         将访问状态标记转换为布尔类型。
#
#         如果访问状态已经是布尔类型，则直接返回；否则，需要转换为布尔类型。
#         这种转换是为了确保访问状态的类型一致性，便于后续逻辑判断和处理。
#
#         Returns:
#             torch.Tensor: 表示访问状态的布尔类型张量。
#         """
#         # 检查访问状态的类型，如果是布尔类型，则直接返回。
#         if self.visited_.dtype == torch.bool:
#             return self.visited_
#         # 如果不是布尔类型，则通过mask_long2bool函数转换为布尔类型，并返回。
#         else:
#             return mask_long2bool(self.visited_, n=self.loc.size(-2))
#
#     def __getitem__(self, key):
#         """
#         重写 __getitem__ 方法，用于按照给定的键访问StateTSP实例中的元素。
#         此方法支持通过张量或切片访问，以便于高效地批量处理数据。
#
#         参数:
#         key: 访问StateTSP实例的键，可以是张量或切片对象。
#
#         返回:
#         如果键是张量或切片，返回一个新的StateTSP实例，其中包含按键选择的元素；
#         否则，调用并返回父类的__getitem__方法的结果。
#         """
#         # 判断key是否为张量或切片，用于批量处理
#         if torch.is_tensor(key) or isinstance(key, slice):
#             # 如果是，通过张量或切片索引所有相关属性，并返回一个新的StateTSP实例。
#             return self._replace(
#                 ids=self.ids[key],
#                 first_a=self.first_a[key],
#                 prev_a=self.prev_a[key],
#                 visited_=self.visited_[key],
#                 lengths=self.lengths[key],
#                 cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
#             )
#         # 如果key不是张量或切片，调用父类的__getitem__方法。
#         return super(StateTSP, self).__getitem__(key)
#
#     @staticmethod
#     def initialize(loc, visited_dtype=torch.bool):
#         """
#         初始化TSP问题的状态.
#
#         参数:
#         - loc: 一个形状为(batch_size, n_loc, 2)的张量，表示每个位置的坐标.
#         - visited_dtype: 用于visited标记的数据类型，可以是torch.bool或其他整型.
#
#         返回:
#         - 一个StateTSP对象，包含了初始化的状态信息.
#         """
#         # 获取批次大小、位置数量和坐标维度
#         batch_size, n_loc, _ = loc.size()
#         # 初始化上一个动作张量为全零，形状为(batch_size, 1)，数据类型为长整型
#         prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
#         # 根据visited_dtype的不同，使用不同的方式存储visited信息
#         visited_ = torch.zeros(
#             batch_size, 1, n_loc, dtype=torch.bool, device=loc.device
#         ) if visited_dtype == torch.bool else torch.zeros(
#             batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device
#         )
#         # 初始化长度张量为全零，形状为(batch_size, 1)
#         lengths = torch.zeros(batch_size, 1, device=loc.device)
#         # 初始化步数张量为全零，形状为(1,)
#         i = torch.zeros(1, dtype=torch.int64, device=loc.device)
#
#         # 返回StateTSP对象，包含初始化的状态信息
#         return StateTSP(
#             loc=loc,
#             ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
#             first_a=prev_a,
#             prev_a=prev_a,
#             visited_=visited_,
#             lengths=lengths,
#             cur_coord=None,
#             i=i
#         )
#
#     def get_final_cost(self):
#         """
#         计算并返回最终成本。
#
#         此函数首先断言所有任务已完成，并且某个条件成立（虽然相关断言代码已被注释掉）。
#         然后，它根据当前坐标和任务的起始点计算出最终成本。
#         最终成本包括两部分：已完成任务的长度和从当前位置到下一个任务起点的直线距离。
#
#         注意：部分代码（如第二个断言）被注释掉，可能是因为某个条件的检查被暂时禁用或移除。
#         """
#         # 断言所有任务已完成。这确保了计算最终成本的前提条件被满足。
#         assert self.all_finished()
#         # 注意：下面的断言检查某个条件，但该条件的具体含义和作用未从当前代码中体现。
#         # assert self.visited_.
#
#         # 计算并返回最终成本。包括已完成任务的长度和到下一个任务起点的直线距离。
#         return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)
#
#
#     def addmask(self):
#         """
#         该方法用于更新对象的visited_属性，通过在特定位置散播值来标记已访问状态。
#
#         方法通过在内部计算并更新visited_的值来实现，主要应用于对象状态的内部管理，特别是与“访问”状态相关的操作。
#
#         Returns:
#             更新后的对象: 通过散播操作更新visited_属性，返回更新后的对象实例。
#         """
#         # 更新visited_属性，使用scatter方法在first_a指定的位置上将值设置为1，以标记这些位置为已访问
#         visited_ = self.visited_.scatter(-1, self.first_a[:, :, None], 1)
#         # 使用更新后的visited_值替换当前对象的对应属性，并返回更新后的对象实例
#         return self._replace(visited_=visited_)
#
#     def update(self, selected, mat, input):
#         """
#         更新模型的隐藏状态。
#
#         参数:
#         - selected: 选定的动作，形状为 (batch_size, 1) 的张量。
#         - mat: 模型矩阵，包含特定于模型的参数。
#         - input: 输入张量，形状为 (batch_size, length)。
#
#         返回:
#         - 更新后的模型隐藏状态实例。
#         """
#         # 更新动作状态
#         prev_a = selected[:, None]  # 为步骤添加一个维度
#         # 计算选定动作对状态的影响
#         _,ind = torch.max(input, dim=2)
#         bdd = mat.var[self.prev_a.squeeze() * mat.n_c + prev_a.squeeze()].unsqueeze(1)
#         add = mat.__getd__(ind, self.prev_a, prev_a, self.lengths).unsqueeze(1)
#         # 根据随机分布和动作影响调整状态
#         bdd = torch.randn(prev_a.size(0), 1, device=device) * bdd
#         bdd = bdd.repeat(1, 2)
#         bdd[:, 1] = add.squeeze() * 5
#         bdd = torch.min(bdd, dim=1)[0]
#         bdd = bdd[:, None].repeat(1, 2)
#         bdd[:, 1] = add.squeeze() * -0.9
#         bdd = torch.max(bdd, dim=1)[0]
#         # 根据状态变化更新长度
#         lengths = self.lengths + add + bdd[:, None]
#         # 记录已访问的区域
#         visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
#         # 返回更新后的模型隐藏状态
#         return self._replace(prev_a=prev_a, visited_=visited_, lengths=lengths, i=self.i + 1)
#
#     def all_finished(self):
#         """
#         判断是否所有步骤都已完成。
#
#         返回:
#             bool: 如果所有步骤都已完成，则为True；否则为False。
#         """
#         # 检查当前步骤是否大于等于总步骤数减1，用于确定是否完成所有步骤
#         return self.i.item() >= self.loc.size(-2) - 1
#
#
#     def get_current_node(self):
#         """
#         获取当前节点。
#
#         此方法用于返回链表中的当前节点。在链表的上下文中，当前节点通常是通过指针指向的，
#         这里返回的是前一个节点（prev_a），这表明在特定的链表实现中，prev_a 可能代表了当前节点的前一个节点，
#         而当前节点可能是指向最新添加的节点或者头部节点等，具体取决于链表的实现方式。
#
#         返回:
#             当前节点（prev_a）的引用。
#         """
#         return self.prev_a
#
#
#     def get_mask(self):
#         """
#         获取当前对象的访问掩码。
#
#         返回:
#             self.visited_: 当前对象的访问掩码属性。这是一个表示哪些元素已被访问的标记数组。
#         """
#         return self.visited_
#
# class AttentionModelFixed(NamedTuple):
#     """
#         注意力模型解码器的上下文，解码过程中固定不变，因此可以预先计算或缓存
#         本类支持同时对多个张量进行高效索引
#     """
#     # 节点嵌入，用于表示图中每个节点的特征
#     node_embeddings: torch.Tensor
#     # 节点上下文的投影，用于注意力机制中的键值对计算
#     context_node_projected: torch.Tensor
#     # 注意力机制中的查询向量，用于生成对输入的“一瞥”
#     glimpse_key: torch.Tensor
#     # 注意力机制中的值向量，用于生成对输入的“一瞥”
#     glimpse_val: torch.Tensor
#     # 用于最终选择操作的键向量，参与生成最终的未归一化对数概率
#     logit_key: torch.Tensor
#
#     def __getitem__(self, key):
#         """
#         重载获取元素的方法，以支持通过键访问AttentionModelFixed的实例。
#
#         这个方法允许对象以字典或序列的方式被索引，从而提供更灵活的访问方式。
#
#         参数:
#         - key: 用于访问元素的键，可以是tensor或切片对象。
#
#         返回:
#         - 如果key是tensor或切片，返回一个新的AttentionModelFixed实例，其中包含了
#           通过key选择的节点嵌入和上下文节点投影，以及相应的glimpse和logit键值。
#         - 如果key不符合上述类型，调用并返回超类(AttentionModelFixed)的__getitem__方法的结果。
#         """
#         # 判断key是否为tensor或切片，如果是，返回一个新的AttentionModelFixed实例
#         if torch.is_tensor(key) or isinstance(key, slice):
#             # 创建并返回一个新的AttentionModelFixed实例，其中包括了通过key选择的
#             # 节点嵌入、上下文节点投影、glimpse键值和logit键值。
#             return AttentionModelFixed(
#                 node_embeddings=self.node_embeddings[key],
#                 context_node_projected=self.context_node_projected[key],
#                 glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
#                 glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
#                 logit_key=self.logit_key[key]
#             )
#         # 如果key不是tensor或切片，调用超类的__getitem__方法
#         return super(AttentionModelFixed, self).__getitem__(key)
#
#
# class AttentionModel(nn.Module):
#     """
#     初始化AttentionModel类的方法，用于配置模型的参数和组件。
#
#     参数:
#     - embedding_dim: int，嵌入层的维度。
#     - hidden_dim: int，隐藏层的维度。
#     - n_encode_layers: int，编码层的数量，默认为2。
#     - tanh_clipping: float，tanh函数的截断值，默认为10.0。
#     - mask_inner: bool，是否掩码内部节点，默认为True。
#     - mask_logits: bool，是否掩码logits，默认为True。
#     - normalization: str，归一化方法，默认为'batch'。
#     - n_heads: int，多头注意力的头数，默认为8。
#     - checkpoint_encoder: bool，是否在编码器中使用梯度检查点，默认为False。
#     - shrink_size: int，缩减尺寸，默认为None。
#     - input_size: int，输入尺寸，默认为4。
#     - max_t: int，最大时间步数，默认为12。
#     """
#
#     def __init__(self,
#                  embedding_dim,
#                  hidden_dim,
#                  n_encode_layers=2,
#                  tanh_clipping=10.,
#                  mask_inner=True,
#                  mask_logits=True,
#                  normalization='batch',
#                  n_heads=8,
#                  checkpoint_encoder=False,
#                  shrink_size=None,
#                  input_size=4,
#                  max_t=12):
#
#         # 调用父类的初始化方法。
#         super(AttentionModel, self).__init__()
#
#         # 设置模型参数。
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.n_encode_layers = n_encode_layers
#         self.decode_type = None
#         self.temp = 1.0
#         self.tanh_clipping = tanh_clipping
#         self.mask_inner = mask_inner
#         self.mask_logits = mask_logits
#         self.n_heads = n_heads
#
#         # 定义必要的组件。
#         step_context_dim = 4 * embedding_dim  # 嵌入首尾节点
#         node_dim = 100
#         self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
#         self.W_placeholder.data.uniform_(-1, 1)  # 占位符应在激活范围之内
#
#         self.init_embed = nn.Linear(node_dim, embedding_dim)
#
#         self.embedder = GraphAttentionEncoder(
#             n_heads=n_heads,
#             embed_dim=embedding_dim,
#             n_layers=self.n_encode_layers,
#             normalization=normalization
#         )
#
#         # 对于每个节点计算（glimpse key, glimpse value, logit key），因此是3 * embedding_dim
#         self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
#         self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
#         self.embed_static_traffic = nn.Linear(node_dim * max_t, embedding_dim)
#         self.embed_static = nn.Linear(2 * embedding_dim, embedding_dim)
#
#         # 确保embedding_dim可以被n_heads整除。
#         assert embedding_dim % n_heads == 0
#         # 注意：n_heads * val_dim == embedding_dim，因此project_out的输入是embedding_dim
#         self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
#         self.project_traffic = nn.Linear(input_size * input_size, embedding_dim, bias=False)
#         self.project_visit = nn.Linear(input_size, embedding_dim, bias=False)
#
#         # 预先计算的坐标张量，用于输入。
#         self.xx = torch.tensor([[i for j in range(input_size)] for i in range(input_size)], device=device).view(1, input_size, input_size)
#         self.yy = torch.tensor([[j for j in range(input_size)] for i in range(input_size)], device=device).view(1, input_size, input_size)
#
#     def set_decode_type(self, decode_type, temp=None):
#         """
#         设置解码类型，并可选地设置温度值。
#
#         此方法主要用于配置解码过程中的类型和温度参数。调用时，必须指定解码类型，
#         而温度参数则是可选的，只有在明确提供时才会更新。
#
#         参数:
#         - decode_type: 必需，指定解码类型，用于确定解码过程的模式。
#         - temp: 可选，指定温度值，用于调整解码过程的灵敏度。如果不提供，将保持当前温度值不变。
#         """
#         # 设置解码类型
#         self.decode_type = decode_type
#         # 如果提供了温度值，则更新温度参数
#         if temp is not None:
#             self.temp = temp
#
#     def forward(self, mat, input, return_pi=False):
#         """
#         模型的前向传播过程。
#
#         :param mat: 表示图结构的矩阵。
#         :param input: 形状为 (batch_size, graph_size, node_dim) 的输入节点特征，或包含多个张量的字典。
#         :param return_pi: 是否返回输出序列，这是可选的，因为与使用 DataParallel 不兼容，
#                           结果在不同的 GPU 上可能长度不同。
#         :return: 包含成本、对数似然度以及（可选地）输出序列 (pi) 的元组。
#         """
#         # 根据输入特征初始化节点嵌入。
#         x = self._init_embed(input)
#         # 准备静态矩阵以进行聚集操作，并调整其形状以适应批次大小。
#         z = mat.mat.view(1, 100, 1200).repeat(input.size(0), 1, 1)
#         # 在节点维度上找到最大值的索引，用于后续的聚集操作。
#         _, ind = torch.max(input, dim=2)
#         # 根据最大值的索引从静态矩阵中收集相关的嵌入信息。
#         tr = z.gather(1, ind.view(input.size(0), -1, 1).expand(input.size(0), input.size(1), 1200))
#         # 嵌入收集到的静态交通信息。
#         y = self.embed_static_traffic(tr)
#         # 将初始化的节点嵌入与静态交通嵌入结合，并通过嵌入器。
#         embeddings, _ = self.embedder(self.embed_static(torch.cat((x, y), dim=2)))
#         self.embeddings = embeddings
#         # 内部模型计算，产生对数概率、输出序列 (pi) 和内部状态。
#         _log_p, pi, state = self._inner(input, embeddings, mat)
#
#         # 计算每个序列的成本和掩码。
#         cost, mask = get_costs(input, pi, state, mat)
#         # 对数似然度在模型内部计算，因为按动作返回它与 DataParallel 不兼容，
#         # 因为序列可能在不同的 GPU 上有不同的长度。
#         ll = self._calc_log_likelihood(_log_p, pi, mask)
#         # 返回成本、对数似然度以及（可选地）输出序列。
#         return cost, ll, pi
#
#
#     def _init_embed(self, x):
#         """
#         初始化嵌入操作。
#
#         该方法用于对输入数据进行初始嵌入处理。具体来说，它调用了当前类中定义的`init_embed`方法来完成实际的嵌入操作。
#
#         参数:
#         x -- 输入数据，类型为任意，具体取决于`init_embed`方法的定义。
#
#         返回:
#         返回`init_embed`方法处理后的结果。
#         """
#         return self.init_embed(x)
#     def _precompute(self, embeddings, num_steps=1):
#         """
#         预计算固定上下文和节点嵌入投影以提高效率。
#
#         此方法旨在通过预计算在整个处理步骤中保持不变的投影来增强计算效率。
#         具体来说，它计算图嵌入的固定上下文，并为注意力机制投影节点嵌入。
#
#         参数:
#         - embeddings: 形状为 (batch_size, num_nodes, embed_dim) 的张量，包含初始节点嵌入。
#         - num_steps: 整数，计划进行的注意力步骤数量。用于相应地预计算投影。
#
#         返回:
#         初始化为预计算值的 AttentionModelFixed 实例，准备进行注意力步骤。
#         """
#
#         # 仅为了效率而计算一次图嵌入的固定上下文投影
#         graph_embed = embeddings.mean(1)
#         # 固定上下文 = (batch_size, 1, embed_dim)，以便与并行时间步长进行广播
#         fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
#
#         # 一次性提前计算用于注意力的节点嵌入投影
#         glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
#             self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
#
#         # 由于只有一个头，无需重新排列logit键
#         fixed_attention_node_data = (
#             self._make_heads(glimpse_key_fixed, num_steps),
#             self._make_heads(glimpse_val_fixed, num_steps),
#             logit_key_fixed.contiguous()
#         )
#         # 返回一个为指定步骤数准备好的 AttentionModelFixed 实例
#         return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
#
#
#     def _inner(self, input, embeddings, mat):
#         """
#         此函数是模型的核心部分，执行一系列计算步骤以输出每一步的日志概率、选择节点的序列以及最终状态。
#
#         参数:
#         - input: 模型的输入数据，用于初始化状态。
#         - embeddings: 输入数据的嵌入表示，用于计算日志概率。
#         - mat: 图的邻接矩阵，用于更新状态和计算日志概率。
#
#         返回值:
#         - outputs: 每一步的日志概率，堆叠成一个张量。
#         - sequences: 每一步选择的节点序列，堆叠成一个张量。
#         - state: 完成所有计算步骤后的最终状态。
#         """
#
#         # 初始化列表来存储每一步的输出和选择的节点序列
#         outputs = []
#         sequences = []
#
#         # 初始化旅行商问题（TSP）的状态
#         state = StateTSP.initialize(input)
#         state = state.addmask()
#
#         # 预先计算在每一步中可重用的键和值
#         batch_size = state.ids.size(0)
#
#         # 执行解码步骤直到所有序列完成
#         fixed = self._precompute(embeddings)
#         while not (state.all_finished()):
#
#             # 计算当前步的日志概率和掩码
#             log_p, mask = self._get_log_p(fixed, state, mat, input)
#
#             # 根据计算出的概率和掩码选择序列中的下一个节点
#             selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # 去除步骤维度
#
#             # 根据选择的节点更新状态
#             state = state.update(selected, mat, input)
#
#             # 存储当前步的输出和选择的序列
#             outputs.append(log_p[:, 0, :])
#             sequences.append(selected)
#
#         # 将收集到的列表堆叠成张量并返回
#         return torch.stack(outputs, 1), torch.stack(sequences, 1), state
#
#     def _precompute(self, embeddings, num_steps=1):
#         """
#         预计算固定上下文和节点嵌入投影以提高效率。
#
#         此方法旨在通过预计算在整个处理步骤中保持不变的投影来增强计算效率。
#         具体来说，它计算图嵌入的固定上下文，并为注意力机制投影节点嵌入。
#
#         参数:
#         - embeddings: 形状为 (batch_size, num_nodes, embed_dim) 的张量，包含初始节点嵌入。
#         - num_steps: 整数，计划进行的注意力步骤数量。用于相应地预计算投影。
#
#         返回:
#         初始化为预计算值的 AttentionModelFixed 实例，准备好进行注意力步骤。
#         """
#
#         # 仅为了效率而计算一次图嵌入的固定上下文投影
#         graph_embed = embeddings.mean(1)
#         # 固定上下文 = (batch_size, 1, embed_dim)，以便与并行时间步长进行广播
#         fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
#
#         # 提前计算用于注意力的节点嵌入投影
#         glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
#             self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)
#
#         # 由于只有一个头，无需重新排列用于logit的键
#         fixed_attention_node_data = (
#             self._make_heads(glimpse_key_fixed, num_steps),
#             self._make_heads(glimpse_val_fixed, num_steps),
#             logit_key_fixed.contiguous()
#         )
#         # 返回为指定步骤数准备好的 AttentionModelFixed 实例
#         return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
#
#     def _make_heads(self, v, num_steps=None):
#         """
#         将输入张量v转换为适合多头注意力机制的形状。
#
#         该函数首先校验num_steps的合法性，然后对输入v进行形状变换，以便能够将其拆分成多个“头”（head），
#         每个头将能够并行处理不同的注意力机制。这一步是实现多头注意力机制的关键。
#
#         参数:
#         - v: 输入张量，其形状应与模型的输入维度兼容。
#         - num_steps: 可选参数，用于指定输出张量在“步数”维度上的长度。如果未指定，则假定v的第二个维度为1或num_steps。
#
#         返回值:
#         - 返回一个张量，其形状为(n_heads, batch_size, num_steps, graph_size, head_dim)，
#           适用于后续的多头注意力计算。
#         """
#         # 校验num_steps的合法性，确保其未指定时，v的第二个维度为1或num_steps
#         assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
#
#         # 对输入v进行形状变换，使其能够被拆分成多个头
#         # 首先，调整v的形状，以便能够将其拆分成多个头
#         # 然后，根据num_steps是否指定，调整输出张量在“步数”维度上的长度
#         # 最后，调整维度顺序，以适应多头注意力机制的计算需求
#         return (
#             v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
#             .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
#             .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
#         )
#
#     def _get_log_p(self, fixed, state, mat, input, normalize=True):
#         """
#         计算日志概率分布 `log_p`，用于指导下一步操作。
#
#         参数:
#         - fixed: 包含固定参数的对象，例如上下文节点的嵌入。
#         - state: 当前的状态对象，包含当前的上下文信息等。
#         - mat: 关系矩阵，表示节点之间的关系。
#         - input: 输入数据，用于计算查询向量。
#         - normalize: 布尔值，决定是否对 `log_p` 进行归一化，默认为 `True`。
#
#         返回:
#         - log_p: 未归一化的日志概率分布。
#         - mask: 掩码，用于指示哪些节点是有效的。
#         """
#
#         # 计算查询向量（query），即上下文节点的嵌入
#         query = fixed.context_node_projected + \
#                 self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, mat, input))
#
#         # 计算节点的键（keys）和值（values）
#         glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
#
#         # 计算掩码
#         mask = state.get_mask()
#
#         # 计算未归一化的日志概率分布（logits）
#         log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
#
#         if normalize:
#             # 归一化日志概率分布
#             log_p = F.log_softmax(log_p / self.temp, dim=-1)
#
#         # 确保没有 `NaN` 值
#         assert not torch.isnan(log_p).any()
#
#         return log_p, mask
#
#     def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
#         """
#         计算基于一个到多个关系的逻辑值。
#
#         参数:
#         - query: 查询张量，形状为(batch_size, num_steps, embed_dim)。
#         - glimpse_K: Key张量用于生成瞥视，形状为(batch_size, num_steps, embed_dim)。
#         - glimpse_V: Value张量用于生成瞥视，形状为(batch_size, num_steps, embed_dim)。
#         - logit_K: 用于计算最终逻辑值的Key张量，形状为(batch_size, num_steps, embed_dim)。
#         - mask: 用于掩码的布尔张量，形状与query相同。
#
#         返回:
#         - logits: 计算得到的逻辑值张量，形状为(batch_size, num_steps, graph_size)。
#         - glimpse: 更新后的查询张量，形状为(batch_size, num_steps, embedding_dim)。
#         """
#         # 获取batch大小、序列长度和嵌入维度
#         batch_size, num_steps, embed_dim = query.size()
#         # 计算键和值的维度
#         key_size = val_size = embed_dim // self.n_heads
#
#         # 计算瞥视Query，重新排列维度以便于后续计算
#         glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
#
#         # 批量矩阵乘法计算兼容性
#         compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
#         # 如果内部掩码启用，则应用掩码
#         if self.mask_inner:
#             assert self.mask_logits, "Cannot mask inner without masking logits"
#             compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
#
#         # 批量矩阵乘法计算头部
#         heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)
#
#         # 项目映射得到更新后的查询/瞥视（batch_size, num_steps, embedding_dim）
#         glimpse = self.project_out(
#             heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
#
#         # 最终查询等于瞥视，这一步骤现在不需要，因为可以合并到project_out中
#         # final_Q = self.project_glimpse(glimpse)
#         final_Q = glimpse
#
#         # 批量矩阵乘法计算逻辑值
#         logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
#
#         # 通过裁剪、掩码和softmax从逻辑值计算概率
#         if self.tanh_clipping > 0:
#             logits = F.tanh(logits) * self.tanh_clipping
#         if self.mask_logits:
#             logits[mask] = -math.inf
#
#         return logits, glimpse.squeeze(-2)
#     def _select_node(self, probs, mask):
#         """
#         根据不同的解码类型（greedy或sampling）选择节点。
#
#         参数：
#         probs: Tensor，表示每个节点的概率分布。
#         mask: Tensor，表示不可选节点的掩码。
#
#         返回：
#         selected: Tensor，表示选中的节点索引。
#         """
#         # 确保概率值中没有NaN
#         assert (probs == probs).all(), "Probs should not contain any nans"
#
#         # 根据解码类型选择节点
#         if self.decode_type == "greedy":
#             # 贪婪解码：选择概率最大的节点
#             _, selected = probs.max(1)
#             # 确保选择的节点不是被mask的节点
#             assert not mask.gather(1, selected.unsqueeze(
#                 -1)).data.any(), "Decode greedy: infeasible action has maximum probability"
#
#         elif self.decode_type == "sampling":
#             # 采样解码：从概率分布中采样选择节点
#             selected = probs.multinomial(1).squeeze(1)
#
#             # 检查采样是否正确，由于GPU上的bug可能导致错误
#             # 详情见链接：https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
#             while mask.gather(1, selected.unsqueeze(-1)).data.any():
#                 print('Sampled bad values, resampling!')
#                 selected = probs.multinomial(1).squeeze(1)
#
#         else:
#             # 如果解码类型未知，则抛出异常
#             assert False, "Unknown decode type"
#         return selected
#     def _get_parallel_step_context(self, embeddings, state, mat, input):
#         """
#         返回每一步的上下文信息，可选择性地一次性为多步计算（为了高效评估模型）
#
#         :param embeddings: (batch_size, 图大小, 嵌入维度)
#         :param state: 模型当前的状态，包括已访问节点的信息和动作历史
#         :param mat: 与交通状况相关的矩阵，用于计算当前的交通上下文
#         :param input: 输入张量，用于确定当前动作
#         :return: (batch_size, 步数, 上下文维度)
#         """
#         # 从embeddings获取批次大小和图大小
#         b_s, i_s = embeddings.size(0), embeddings.size(1)
#         # 获取最大值的索引，表示当前动作
#         _, ind = torch.max(input, dim=2)
#         # 根据动作、交通矩阵以及模型状态计算当前的交通上下文
#         current_traffic = self.project_traffic(mat.__getddd__(ind, self.xx.repeat(b_s, 1, 1).view(b_s, i_s*i_s), self.yy.repeat(b_s, 1, 1).view(b_s, i_s*i_s), state.lengths).view(b_s, 1, i_s*i_s))
#         # 计算已访问节点的上下文
#         current_visit = self.project_visit(state.visited_.float())
#         # 收集先前动作和第一个动作的嵌入
#         ss = embeddings.gather(1, torch.cat((state.first_a, state.prev_a), 1)[:, :, None].expand(b_s, 2, embeddings.size(-1)))
#         # 拼接先前动作的嵌入、交通上下文和已访问节点的上下文
#         return torch.cat((ss.view(b_s, 1, -1), current_traffic, current_visit), dim=2)
#
#
#     def _get_attention_node_data(self, fixed, state):
#         """
#         获取注意力机制节点的数据。
#
#         该方法用于从固定的注意力机制参数中提取出当前状态所需的节点数据，
#         包括用于一瞥的键、值，以及用于计算logit的键。
#
#         参数:
#         - fixed: 注意力机制的固定参数，包含glimpse_key、glimpse_val和logit_key等属性。
#         - state: 当前的状态，此处未使用，保留以示例化在其他场景下可能的使用。
#
#         返回值:
#         - 返回一个元组，包括用于一瞥的键（glimpse_key）、用于一瞥的值（glimpse_val），
#           以及用于计算logit的键（logit_key）。
#         """
#         # 从fixed对象中提取并返回所需的节点数据
#         return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
#
#     def _calc_log_likelihood(self, _log_p, a, mask):
#         """
#         计算选定动作的对数似然概率。
#
#         :param _log_p: 所有动作的概率分布的对数值
#         :type _log_p: 张量(Tensor)
#         :param a: 选定动作的索引
#         :type a: 张量(Tensor)
#         :param mask: 一个可选的掩码，用于忽略某些与目标无关的动作
#         :type mask: 张量(Tensor) 或 None
#         :return: 选定动作的对数似然概率
#         :rtype: 张量(Tensor)
#         """
#         # 获取对应于选定动作的 log_p 值
#         log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
#
#         # 可选：通过掩码忽略与目标无关的动作，使其在强化学习过程中不被考虑
#         if mask is not None:
#             log_p[mask] = 0
#
#         # 确保对数概率不是 -inf，因为这可能表明采样过程存在问题
#         # "对数概率不应该为 -inf，请检查采样过程!"
#         assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
#         # 计算 log_likelihood
#         return log_p.sum(1)
