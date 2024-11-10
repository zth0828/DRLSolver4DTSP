import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
from tqdm import tqdm
from torch.utils.data import DataLoader
import copy
from itertools import permutations
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def set_decode_type(model, decode_type):
    """
    设置模型的解码类型。

    本函数的主要作用是调整模型的解码策略，以适应不同的使用场景。
    例如，这可能涉及到在贪婪解码和束搜索解码之间进行切换。

    参数：
    model: 模型对象，具有设置解码类型的方法。
    decode_type: 字符串，指定模型将要采用的解码策略。
    """
    model.set_decode_type(decode_type)
class TSPDataset(Dataset):
    
    def __init__(self, ci, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        """
        初始化TSP数据集。

        参数:
        - ci: 城市信息对象，包含城市数量和坐标。
        - filename: 数据集文件名（未使用）。
        - size: 每个样本中城市的数量。
        - num_samples: 生成的样本数量。
        - offset: 样本索引的偏移量（未使用）。
        - distribution: 城市位置的分布方式（未使用）。
        """
        super(TSPDataset, self).__init__()

        # 初始化数据集为空列表
        self.data_set = []

        # 生成num_samples个随机样本，每个样本包含ci.n_cities-1个城市的位置
        l = torch.rand((num_samples, ci.n_cities - 1))
        # 对每个样本的城市位置进行排序
        sorted, ind = torch.sort(l)
        # 扩展索引维度，以便能够选择城市位置
        ind = ind.unsqueeze(2).expand(num_samples, ci.n_cities - 1, 2)
        # 选择前size个城市的索引，并加1以避免重复
        ind = ind[:,:size,:] + 1
        # 复制ci.cities以匹配样本数量
        ff = ci.cities.unsqueeze(0)
        ff = ff.expand(num_samples, ci.n_cities, 2)
        # 根据索引选择对应的城市位置
        f = torch.gather(ff, dim = 1, index = ind)
        # 调整维度顺序以便处理
        f = f.permute(0,2,1)
        # 将起始点（depot）添加到每个样本中
        depot = ci.cities[0].view(1, 2, 1).expand(num_samples, 2, 1)
        self.static = torch.cat((depot, f), dim = 2)
        # 准备索引，以便在数据集中正确放置城市位置
        depot = torch.zeros(num_samples, 1, 1, dtype=torch.long)
        ind = ind[:,:,0:1]
        ind = torch.cat((depot, ind), dim=1)

        # 初始化数据集矩阵
        self.data = torch.zeros(num_samples, size+1, ci.n_cities)
        # 将城市位置放入数据集矩阵中
        self.data = self.data.scatter_(2, ind, 1.)
        # 记录数据集大小
        self.size = len(self.data)

    def __len__(self):
        """
        返回实例的大小。

        此方法使得实例能够像Python中的序列或其他可测量长度的对象一样，
        通过内置函数len()获取其大小。

        返回:
            int: 实例的大小。
        """
        return self.size


    def __getitem__(self, idx):
        """
        通过索引访问数据。

        该方法允许类的实例通过索引访问其内部数据，类似于列表或其他序列类型的行为。

        参数:
        idx (int): 访问数据的索引位置。

        返回:
        通过指定索引访问的数据项。
        """
        return self.data[idx]
def get_inner_model(model):
    """
    返回传入的模型对象本身。

    此函数的目的是提供一个通用的接口，使得外部可以统一处理获取模型的操作，
    而不必关心模型的具体实现细节。它主要用于框架或库中，需要操作模型但不
    直接依赖于特定模型实现的场景。

    参数:
    model: 任何模型对象。该函数不关心模型的具体类型或实现，只要求传入的是一个
           模型实例。

    返回值:
    返回传入的模型对象本身。该函数不会对模型进行任何修改或包装，直接返回输入的
    模型对象。
    """
    return model
def rollout(mat, model, dataset, opts):
    """
    使用贪婪解码评估模型在数据集上的性能。

    此函数将模型设置为评估模式，并采用贪婪解码生成序列，然后计算模型在数据集上的性能指标。

    参数:
    - mat: 图的邻接矩阵，用于表示元素之间的依赖结构。
    - model: 需要进行评估的训练好的模型。
    - dataset: 用于评估模型性能的数据集。
    - opts: 包含设备信息和评估时的批大小等选项配置。

    返回:
    - 数据集上模型的性能指标，以拼接后的张量形式返回。
    """
    # 设置模型在评估时使用贪婪解码
    set_decode_type(model, "greedy")
    # 将模型切换到评估模式
    model.eval()

    def eval_model_bat(bat):
        """
        在一批数据上评估模型。

        返回:
        - 该批数据的成本张量，从计算图中分离后移动到CPU。
        """
        # 禁用梯度计算，减少内存消耗并加快计算速度
        with torch.no_grad():
            # 前向传播计算该批数据的成本
            cost, _, _ = model(mat, move_to(bat, opts.device))
        # 返回成本数据，移动到CPU
        return cost.data.cpu()

    # 使用DataLoader加载数据集中的批次数据，并对每个批次进行评估
    return torch.cat([
        eval_model_bat(bat)
        for bat in DataLoader(dataset, batch_size=opts.eval_batch_size)
    ], 0)
def move_to(var, device):
    """
    将变量或变量字典移动到指定的设备上。

    如果输入是一个字典，则递归地将字典中的每个值移动到指定设备。
    如果输入是其他类型的变量，则直接将其移动到指定设备。

    参数:
    var: 待移动的变量，可以是任何类型的变量或变量字典。
    device: 目标设备，用于指定变量移动到的设备。

    返回:
    移动到指定设备后的变量或变量字典。
    """
    # 如果var是一个字典，则递归地处理字典中的每个值
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    # 否则，直接将变量移动到指定设备
    return var.to(device)
class Baseline(object):
    """
    Baseline类作为基础的数据处理和模型评估框架。它提供了一些基本的方法用于数据集包装、批次解包和模型评估等操作。
    这个类的目的是为了定义一个基本的接口，供其他具体的模型类实现或覆盖。
    """

    def wrap_dataset(self, dataset):
        """
        将数据集包装成框架所需格式。

        参数:
        dataset: 要处理的数据集。

        返回:
        直接返回输入的数据集，不做任何处理。
        """
        return dataset

    def unwrap_batch(self, batch):
        """
        将批次数据解包成模型输入和可能的其他信息。

        参数:
        batch: 一个批次的数据。

        返回:
        一个包含批次数据和None的元组，表示没有额外信息需要处理。
        """
        return batch, None

    def eval(self, x, c):
        """
        对给定的输入进行模型评估。

        参数:
        x: 模型的输入数据。
        c: 可能的条件或上下文信息。

        抛出:
        NotImplementedError: 表示这个方法应该被子类实现。
        """
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        """
        获取模型中可学习的参数列表。

        返回:
        一个空列表，表示该模型没有可学习的参数。
        """
        return []

    def epoch_callback(self, model, epoch):
        """
        在每个训练周期结束时调用的回调函数。

        参数:
        model: 当前的模型对象。
        epoch: 当前的训练周期数。

        该方法不执行任何操作，供子类根据需要实现特定的回调逻辑。
        """
        pass

    def state_dict(self):
        """
        获取模型的当前状态字典。

        返回:
        一个空字典，表示该模型没有状态需要保存。
        """
        return {}

    def load_state_dict(self, state_dict):
        """
        从给定的状态字典加载模型状态。

        参数:
        state_dict: 模型的状态字典。

        该方法不执行任何操作，供子类根据需要实现特定的状态加载逻辑。
        """
        pass
class WarmupBaseline(Baseline):
    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        """
        初始化Baseline类的实例。

        参数:
        - baseline: 一个基线对象，用于在训练过程中进行比较。
        - n_epochs: 整数，指定在指数基线上进行预热的周期数。预热目的是在训练初期逐步调整学习率。
        - warmup_exp_beta: 浮点数，指定指数基线的平滑系数。该参数控制预热过程中学习率的下降速度。
        """
        super(Baseline, self).__init__()
        # 设置外部提供的基线对象，用于后续的性能比较。
        self.baseline = baseline
        # 确保预热周期数为正数，因为至少需要一个周期来进行预热。
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        # 初始化一个指数基线对象用于预热，使用给定的平滑系数。
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        # 初始化alpha值为0，它将在预热过程中逐步增加，用于平衡基线值。
        self.alpha = 0
        # 保存指定的预热周期数，用于控制预热过程的长度。
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        """
        根据当前实例的alpha值，决定使用基线模型还是热身基线模型来包装数据集。

        此方法主要用于处理数据集的预处理或增强，依据alpha值的不同选择不同的处理策略。
        如果alpha大于0，则使用基线模型的处理策略；否则使用热身基线模型的处理策略。

        参数:
        dataset -- 需要被包装的数据集。

        返回:
        返回基线模型或热身基线模型处理过的数据集。
        """
        # 根据alpha值的条件判断，选择不同的数据集处理策略
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)


    def unwrap_batch(self, batch):
        """
        根据当前策略和条件，调用相应的函数来处理batch数据。

        该函数基于alpha值的条件，动态选择使用baseline策略还是warmup_baseline策略来解包batch数据。
        当alpha大于0时，认为是应用了某种策略，因此使用baseline策略处理batch；
        否则，使用warmup_baseline策略处理。

        参数:
        - batch: 待处理的batch数据。

        返回值:
        - 返回baseline或warmup_baseline策略处理后的结果。
        """
        # 根据alpha值判断使用哪种策略
        if self.alpha > 0:
            # 当alpha大于0时，调用baseline策略下的unwrap_batch方法处理batch
            return self.baseline.unwrap_batch(batch)
        # 当alpha不大于0时，调用warmup_baseline策略下的unwrap_batch方法处理batch
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        """
        根据 `alpha` 的值评估性能。

        参数:
        - x: 评估的第一个参数。
        - c: 评估的第二个参数。

        返回:
        - 包含评估结果和损失的一个元组。

        该方法首先检查 `alpha` 是否为 1 或 0，如果是，则分别返回基线或预热基线的评估结果。对于其他 `alpha` 值，它计算两个基线结果的凸组合并返回。
        """
        # 检查 `alpha` 是否为 1，返回基线的评估结果
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        # 检查 `alpha` 是否为 0，返回预热基线的评估结果
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        # 计算基线的评估结果和损失
        v, l = self.baseline.eval(x, c)
        # 计算预热基线的评估结果和损失
        vw, lw = self.warmup_baseline.eval(x, c)
        # 返回基线和损失的凸组合
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        """
        在每个训练周期结束时调用的回调函数。

        该函数的主要目的是更新warmup策略的状态，包括调用内部模型的周期回调，
        以及计算和打印当前周期的warmup alpha值。

        参数:
        - model: 当前正在训练的模型。
        - epoch: 当前已完成的训练周期数。
        """
        # 调用内部模型的周期回调，确保所有必要的周期末操作都能被执行
        self.baseline.epoch_callback(model, epoch)

        # 计算当前周期的warmup alpha值，alpha值随着周期数的增加而增加
        self.alpha = (epoch + 1) / float(self.n_epochs)

        # 如果当前周期小于设定的总周期数，打印当前的warmup alpha值
        if epoch < self.n_epochs:
            print("Set warmup alpha = {}".format(self.alpha))


    def state_dict(self):
        """
        返回基线模型的状态字典。

        在预热阶段进行检查点操作没有意义；此时仅保存内部的基线模型。
        此方法从父类重写以针对特定场景自定义行为。

        返回:
            dict: 基线模型的状态字典。
        """
        # 在预热阶段创建检查点是没有意义的，只保存内部的基线模型
        return self.baseline.state_dict()


    def load_state_dict(self, state_dict):
        """
        加载状态字典到模型的基本部分。此方法主要用于支持模型状态的加载，
        在模型预热阶段加载检查点没有意义，因此仅加载内部基本模型的状态。

        参数:
        - state_dict (dict): 要加载的状态字典，通常从模型检查点加载。
        """
        # 预热阶段内的检查点加载没有意义，只加载内部基线模型
        self.baseline.load_state_dict(state_dict)
class NoBaseline(Baseline):

    """
    一个处理评估指标的类。

    该类提供了计算准确率、精确度、召回率和F1分数等评估指标的方法。
    它不涉及基线或损失的计算，这些通常由系统的其他部分处理。
    """

    def eval(self, x, c):
        """
        计算并返回评估指标。

        参数:
        - x: 输入数据或预测结果。
        - c: 正确标签或比较标准。

        返回:
        - accuracy: 预测结果的准确率得分。
        - loss: 根据预测结果和标签计算出的损失。

        注意: 当前方法返回(0, 0)作为占位符，表示没有进行基线或损失的计算。
        """
        return 0, 0  # 没有基线，没有损失
class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        # 调用父类的初始化方法
        super(Baseline, self).__init__()

        # 初始化beta参数
        self.beta = beta
        # 初始化v变量，具体用途和初始化值未在代码中明确
        self.v = None

    def eval(self, x, c):
        """
        评估函数

        该函数主要用于计算和更新参数的加权平均值。在第一次调用时，如果没有预先设定的v值，
        它将计算传入参数c的平均值作为v。在后续调用中，它将使用一个加权因子beta来更新v的值。

        参数:
        x: 输入数据，此处未直接使用，因此可以忽略。
        c: Tensor类型，用于计算或更新v值的输入。

        返回:
        self.v: Tensor类型，计算得到的加权平均值。
        0: 表示没有损失(loss)，因为此函数中不进行反向传播。
        """
        # 判断是否是第一次调用，如果是，则初始化v值
        if self.v is None:
            v = c.mean()  # 计算c的平均值作为初始v
        else:
            # 使用加权因子beta更新v值
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        # 将计算得到的v值从计算图中分离出来，因为我们不希望对其进行反向传播
        self.v = v.detach()
        # 返回更新后的v值和一个表示没有损失的0值
        return self.v, 0


    def state_dict(self):
        """
        获取实例的状态字典

        本方法用于返回一个包含实例状态信息的字典，以便于序列化或存储实例的状态

        Returns:
            dict: 包含实例状态信息的字典，当前仅包含'v'键对应的实例变量
        """
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        """
        从给定的状态字典中加载数据。

        参数:
            state_dict (dict): 包含状态数据的字典，其中'v'是此方法关注的键。

        返回:
            无
        """
        # 加载状态字典中的'velocity'数据到当前实例
        self.v = state_dict['v']
class CriticBaseline(Baseline):

    def __init__(self, critic):
        # 调用基类的构造函数，进行初始化
        super(Baseline, self).__init__()

        # 初始化Baseline类的实例，设置critic为基线模型
        self.critic = critic

    def eval(self, x, c):
        """
        评估给定输入x和目标值c的模型性能。

        参数:
        x -- 输入数据，用于评估模型的当前状态。
        c -- 目标值，用于比较模型的预测值。

        返回:
        该方法返回一个元组，其中包含模型对输入x的评估值和模型预测与目标值之间的均方误差损失。
        """
        # 使用critic网络评估输入x的价值
        v = self.critic(x)
        # 由于actor网络不应该通过critic网络的输出进行反向传播，所以需要分离v的梯度
        # 这样可以避免在更新actor网络时，对critic网络进行不必要的更新
        # 只有在计算损失时才需要critic网络的输出，因此在此处分离梯度
        return v.detach(), F.mse_loss(v[:,0], c.detach())


    def get_learnable_parameters(self):
        """
        获取可学习的参数列表

        该方法返回critic网络中所有可学习的参数列表，这些参数通常是在优化过程中被更新的权重和偏置。

        Returns:
            list: 包含所有可学习参数的列表
        """
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        """
        在每个训练周期结束时调用的回调函数。

        此函数设计用于在训练过程中的每个周期（epoch）结束时执行特定的操作，
        比如保存模型、记录日志等。默认情况下，此函数不执行任何操作（pass），
        但用户可以根据需要对其进行重写（override）。

        参数:
        - model: 当前正在训练的模型对象，允许用户对模型进行评估或保存等操作。
        - epoch: 当前已完成的训练周期数，用于记录或基于周期数进行特定操作。

        返回值:
        无。此函数不返回任何值。
        """
        pass

    def state_dict(self):
        """
        获取当前实例的状态字典。

        此方法主要用于保存模型的参数状态。它通过返回一个包含模型状态的字典，
        允许我们方便地保存和加载模型，以便在未来的应用中使用。

        Returns:
            dict: 包含模型状态的字典，当前仅包含critic网络的状态。
        """
        # 返回包含critic网络状态的字典
        return {
            'critic': self.critic.state_dict()
        }


    def load_state_dict(self, state_dict):
        """
        加载状态字典到当前实例中。

        此函数旨在从给定的状态字典中加载critic网络的状态。
        它首先尝试从状态字典中获取critic相关的状态字典。
        如果获取到的critic状态字典不是字典类型，则假定它是旧版本的兼容性对象，并调用其state_dict方法转换为字典。
        最后，将critic的状态字典加载到当前的critic网络中，覆盖现有的状态。

        参数:
        - state_dict: 包含模型状态的字典，其中应包含critic网络的状态。

        返回值:
        无返回值。
        """
        # 尝试从状态字典中获取critic相关的状态字典
        critic_state_dict = state_dict.get('critic', {})
        # 如果获取到的critic状态字典不是字典类型，则假定它是旧版本的兼容性对象，并调用其state_dict方法转换为字典
        if not isinstance(critic_state_dict, dict):
            critic_state_dict = critic_state_dict.state_dict()
        # 将critic的状态字典加载到当前的critic网络中，覆盖现有的状态
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})
class RolloutBaseline(Baseline):
    def __init__(self, mat, ci, model, opts, epoch=0):
        super(Baseline, self).__init__()  # 调用基类的构造函数
        self.mat = mat  # 设置数据矩阵
        self.opts = opts  # 设置运行时选项
        self.ci = ci  # 设置类索引
        self.last = 0  # 初始化一个标识符为0，用于后续操作
        self._update_model(model, epoch)  # 更新模型状态，传入当前模型和训练周期数
    def _update_model(self, model, epoch, dataset=None):
        """
        更新模型并生成基线数据集。

        在更新模型时总是生成基线数据集，以防止模型过度拟合到基线数据集上。

        参数:
        - model: 要更新的模型。
        - epoch: 当前的训练周期数。
        - dataset: 用于验证的基线数据集（默认为None）。

        如果提供的数据集与配置的验证大小或图形大小不匹配，则不使用该数据集。
        如果数据集不匹配，将生成一个新的基线数据集。

        该方法不返回值，但更新类的内部状态，包括模型、数据集以及基线模型的评估结果。
        """
        # 深拷贝模型以保留当前状态
        self.model = copy.deepcopy(model)
        # 当更新模型时，总是生成基线数据集以防止过拟合
        if dataset is not None:
            # 检查数据集大小是否与配置的验证大小匹配
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            # 检查数据集的第一个样本的大小是否与配置的图形大小匹配
            elif dataset[0].size(0) != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None
        # 如果数据集不匹配或未提供，则生成新的基线数据集
        if dataset is None:
            self.dataset = TSPDataset(self.ci, size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution)
        else:
            self.dataset = dataset
        # 在评估数据集上评估基线模型并更新内部状态
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.mat, self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch


    def _reload_model(self, model):
        # 深拷贝当前模型到新的model变量中，确保重新加载的模型与原模型完全独立
        model = copy.deepcopy(self.model)
        # 输出提示信息，表明模型已重新加载
        print('reloaded')

    def wrap_dataset(self, dataset):
        """
        将给定的数据集包装到BaselineDataset中进行评估。

        参数:
        - dataset: 数据集实例，用于加载数据和标签。

        返回:
        - BaselineDataset实例，包含了原始数据集和经过模型处理的基线数据。

        说明:
        此函数用于将原始数据集和通过模型生成的基线数据包装在一起，以便进行基线评估。
        基线数据需要转换为2D形式，以避免在使用PyTorch DataLoader时自动转换为double类型的问题。
        """
        # 打印开始评估基线的提示信息
        print("Evaluating baseline on dataset...")
        # 需要将基线转换为2D形式，以防止在使用PyTorch DataLoader时自动转换为double类型
        # 参考讨论: https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        return BaselineDataset(dataset, rollout(self.mat, self.model, dataset, self.opts).view(-1, 1))


    def unwrap_batch(self, batch):
        """
        解包数据批次。

        该方法旨在处理数据批次，将数据和基线信息分离出来，其中数据保持原样，而基线信息则被展平为一维结构。

        参数:
        - batch: 一个字典，包含两个键值对：'data' 和 'baseline'，分别代表数据批次中的数据和对应的基线信息。

        返回值:
        - 一个元组，包含两部分：第一部分是原始数据，第二部分是被展平为一维的基线信息。
        """
        # 从批次数据中提取数据和基线信息，并将基线信息展平为一维，以便后续处理
        return batch['data'], batch['baseline'].view(-1)

    def eval(self, x, c):
        """
        评估给定状态的价值而不更新模型参数。

        参数:
        - x: 状态输入，通常是当前状态的张量表示。
        - c: 在此方法中未使用，可以是任何占位符或额外参数。

        返回值:
        - v: 输入状态的评估价值。
        - 0: 由于此方法中没有损失计算，第二个返回值始终为0。
        """
        # 使用volatile模式进行高效的推理（单个批次，因此我们不使用rollout函数）
        with torch.no_grad():
            v, _ = self.model(self.mat, x, self.model1, self.model2)

        # 没有损失
        return v, 0


    def epoch_callback(self, model, epoch):
        """
        在每个训练周期结束时挑战当前基线模型，并如果表现更好则替换基线模型。
        :param model: 用于挑战基线模型的当前模型
        :param epoch: 当前的训练周期
        """
        # 在评估数据集上评估候选模型
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(self.mat, model, self.dataset, self.opts).cpu().numpy()

        # 计算候选模型的平均值
        candidate_mean = candidate_vals.mean()

        # 打印当前周期、候选模型平均值、基线模型周期和平均值以及它们的差异
        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))

        # 如果候选模型的平均值低于基线模型的平均值，则进行统计显著性测试
        if candidate_mean - self.mean < 0:
            # 计算p值
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2  # 单侧检验
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            # 如果p值小于设定的显著性水平，则更新基线模型
            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)
        # 如果候选模型的平均值不低但训练周期已经超过一定数量，则重新加载之前的模型
        elif (epoch - self.epoch > 99 and epoch - self.last > 99):
            self._reload_model(model)
            self.last = epoch

    def state_dict(self):
        """
        获取当前实例的状态字典。

        此方法用于返回一个包含模型状态信息的字典，主要包括模型参数、数据集信息和当前训练周期。
        这对于保存和加载模型状态非常有用，尤其是在训练过程中断点续训时。

        Returns:
            dict: 包含模型状态信息的字典，包括：
            - 'model': 模型参数
            - 'dataset': 数据集信息
            - 'epoch': 当前训练周期
        """
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        """
        加载模型的状态字典，以恢复模型的训练状态。

        此函数设计为无论模型是使用数据并行还是非数据并行方式保存，均能正确加载。
        它首先创建当前模型的深拷贝，然后从状态字典中提取模型参数，最后更新模型、训练周期和数据集信息。

        参数:
        - state_dict: 包含模型状态、训练周期和数据集信息的字典。

        返回值:
        无返回值，但会直接更新模型的状态。
        """
        # 创建当前模型的深拷贝，以安全地加载状态
        load_model = copy.deepcopy(self.model)

        # 提取并加载状态字典中模型的内部状态
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())

        # 更新模型、训练周期和数据集信息
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])
class BaselineDataset(Dataset):
    """
    Baseline数据集类，继承自Dataset类。
    该类主要用于将一个数据集和与之对应的基线数据进行绑定，确保在数据加载时能够同时获取到数据及其对应的基线信息。

    参数:
    - dataset: 要绑定的基础数据集。
    - baseline: 与基础数据集对应的一组基线数据。

    注意: 构造函数中通过断言语句确保了dataset和baseline的长度相同，保证了数据的一致性。
    """

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline)) # "数据集和基线数据的长度必须相同"

    def __getitem__(self, item):
        """
        获取给定索引处的数据及其对应的基线数据。

        参数:
        - item: 要获取数据的索引。

        返回:
        一个字典，包含当前索引处的'data'和'baseline'。
        """
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        """
        返回数据集的大小。

        返回:
        数据集中的数据数量。
        """
        return len(self.dataset)