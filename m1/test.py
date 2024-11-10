import os
import time
import torch
import math
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import numpy as np
from transformer import AttentionModel
from scipy.interpolate import CubicSpline

import torch.optim as optim
from tensorboard_logger import Logger as TbLogger


from options import get_options
from baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
import warnings
import pprint as pp
warnings = warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Cities:
    def __init__(self, n_cities=100):
        """
        初始化问题实例，生成指定数量的城市坐标集合。

        参数:
        - n_cities: int，城市数量，默认为100。表示要在平面内随机生成的城市数目。

        返回:
        无返回值，此方法用于初始化类的实例。
        """
        # 将城市数量保存为实例变量，以便在类的其他方法中使用。
        self.n_cities = n_cities
        # 随机生成n_cities个城市的二维坐标，并保存为实例变量。
        self.cities = torch.rand((n_cities, 2))
    def __getdis__(self,i, j):
        """
        计算两个城市之间的欧氏距离。

        该方法通过计算两个城市坐标差的平方和然后开平方的方式，得到两个城市之间的直线距离。

        参数:
        i (int): 第一个城市的索引。
        j (int): 第二个城市的索引。

        返回:
        torch.Tensor: 两个城市之间的欧氏距离。
        """
        # 使用PyTorch计算两个城市坐标之差的平方和并开平方，得到欧氏距离
        return torch.sqrt(torch.sum(torch.pow(torch.sub(self.cities[i], self.cities[j]), 2)))


class DistanceMatrix:
    def __init__(self, ci, max_time_step = 100, load_dir = None):
        """
        初始化函数

        参数:
        ci: 包含城市数量信息的配置对象
        max_time_step: 最大时间步长，默认为100
        load_dir: 加载数据的目录，如果为None，则不加载，默认为None
        """
        # 初始化城市数量和最大时间步长
        self.n_c = ci.n_cities
        self.max_time_step = max_time_step

        # 在无梯度的上下文中初始化模型参数
        # 这里的参数包括控制不同时间步长下城市间相互作用的矩阵
        with torch.no_grad():
            self.mat = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m2 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m3 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m4 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)

            # 初始化变量，这里是一个包含城市间作用强度的向量
            # 初始值设为0.03，形状为(n_cities * n_cities, 1)
            self.var = torch.full((ci.n_cities * ci.n_cities, 1), 0.03, device = device).view(-1)

            # 如果指定了加载目录，则从该目录加载数据并初始化参数
            if (load_dir is not None):
                # 加载数据，并初始化参数
                temp = np.loadtxt(load_dir, delimiter=',', skiprows=0)
                x = np.arange(max_time_step + 1)
                for k in range(self.n_c):
                    # 对角线元素设为0，表示城市与自身的作用强度为0
                    self.var[k*self.n_c+k] = 0
                    for j in range(self.n_c):
                        # 计算城市间作用的参数
                        i = k * self.n_c + j
                        # 使用三次样条插值计算并赋值
                        cs = CubicSpline(x, np.concatenate((temp[i], [temp[i,0]]), axis=0), bc_type='periodic')
                        self.m4[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[0], device=device)
                        self.m3[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[1], device=device)
                        self.m2[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[2], device=device)
                        self.mat[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[3], device=device)

    def __getd__(self, st, a, b, t):
        """
        根据给定的状态和时间参数，计算特定的加权和。

        参数:
        - st: 输入状态张量。
        - a: 第一个时间点的索引。
        - b: 第二个时间点的索引。
        - t: 时间参数，用于插值计算。

        返回:
        - res: 计算得到的加权和结果。
        """
        # 通过索引a和b从状态张量st中提取对应的时间步状态
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)
        # 计算时间参数t对应的时间步，以及用于插值计算的相邻两个整数时间步
        tt = torch.floor(t * self.max_time_step) % self.max_time_step
        zz = (torch.floor(t * self.max_time_step) + 1) % self.max_time_step
        # 根据状态和时间步计算两个关键值c和d，用于后续从矩阵中提取数据
        c = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + tt.squeeze().long()
        d = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + zz.squeeze().long()
        # 从预定义的矩阵中根据c和d提取相应的值
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2, 0, c)
        a2 = torch.gather(self.m3, 0, c)
        a3 = torch.gather(self.m4, 0, c)
        b0 = torch.gather(self.mat, 0, d)
        # 计算插值参数z及其幂次，用于后续的插值计算
        z = (t.squeeze() * self.max_time_step - torch.floor(t.squeeze() * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z
        # 根据插值参数和提取的矩阵值计算结果
        res = a0 + a1 * z + a2 * z2 + a3 * z3
        # 设置结果的最小值和最大值限制，以保证结果的合理性
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        # 应用最小值限制
        res,_ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim = -1), dim = -1)
        # 应用最大值限制
        res,_ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim = -1), dim = -1)
        return res
    def __getddd__(self, st, a, b, t):
        """
        计算并返回一个张量，该张量是基于输入参数和内部状态通过特定数学运算得到的结果。

        参数:
        - st: 输入状态张量。
        - a, b: 用于计算的索引张量。
        - t: 时间参数，用于时间相关的计算。

        返回:
        - res: 计算得到的张量。
        """
        # 获取张量a的尺寸，用于后续的形状恢复
        s0, s1 = a.size(0), a.size(1)

        # 通过索引从st中提取对应元素，用于后续计算
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)

        # 计算时间参数相关的值，用于确定计算中的特定时间点
        tt = torch.round(t * self.max_time_step) % self.max_time_step
        zz = (torch.round(t * self.max_time_step) + 1) % self.max_time_step

        # 将索引张量转换为一维，准备进行下一步的计算
        c = a * self.n_c * self.max_time_step + b * self.max_time_step + tt.long()
        c = c.view(-1)

        d = a * self.n_c * self.max_time_step + b * self.max_time_step + zz.long()
        d = d.view(-1)

        # 从内部矩阵中提取需要的值
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2, 0, c)
        a2 = torch.gather(self.m3, 0, c)
        a3 = torch.gather(self.m4, 0, c)
        b0 = torch.gather(self.mat, 0, d)

        # 准备时间参数，用于计算插值
        tt = tt.view(-1)
        ttt = t.expand(s0, s1).contiguous().view(-1)
        z = (ttt * self.max_time_step - torch.floor(ttt * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z

        # 执行核心计算，得到初步结果
        res = a0 + a1 * z + a2 * z2 + a3 * z3

        # 确保结果在合理的范围内，设置最小和最大值
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5

        # 最终结果通过最大值和最小值限制，确保不会超出预期范围
        res,_ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim = -1), dim = -1)
        res,_ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim = -1), dim = -1)

        # 将结果恢复为原始的形状，以便于后续使用
        return res.view(s0, s1)
def rollout(mat, model, dataset, opts):
    """
    使用贪心策略对模型进行评估。

    该函数主要用于在给定的测试数据集上评估模型的性能。它通过设置模型为贪心解码模式，
    并在不更新参数的情况下批量处理数据，以确保评估过程的准确性和高效性。

    参数：
    - mat: 输入矩阵，模型的输入数据。
    - model: 待评估的模型。
    - dataset: 用于评估的数据集。
    - opts: 选项字典，包含评估过程所需的配置，如设备类型和批处理大小。

    返回值：
    - 返回模型在数据集上的评估结果，以张量形式返回。
    """
    # 设置模型为贪心解码模式，以进行快速评估
    set_decode_type(model, "greedy")
    # 将模型设置为评估模式
    model.eval()

    def eval_model_bat(bat):
        """
        内部函数，用于对模型进行批量评估。

        参数：
        - bat: 一个批次的数据。

        返回值：
        - 返回该批次数据的评估结果。
        """
        # 在不跟踪梯度的情况下进行模型评估
        with torch.no_grad():
            cost, _, _ = model(mat, move_to(bat, opts.device))
        # 将评估结果从设备移动到CPU上
        return cost.data.cpu()

    # 对整个数据集进行批处理评估
    return torch.cat([
        eval_model_bat(bat)
        for bat in DataLoader(dataset, batch_size=opts.eval_batch_size)
    ], 0)

def roll(mat, model, dataset, opts):
    """
    使用贪心策略对模型进行评估。

    参数:
    - mat: 输入数据矩阵。
    - model: 待评估的模型。
    - dataset: 用于评估的数据集。
    - opts: 评估选项，包括设备和批处理大小等。

    返回:
    - p: 模型的预测结果堆叠成的张量。
    - c: 模型的代价结果堆叠成的张量。
    """
    # 设置模型为贪心解码类型，用于评估
    set_decode_type(model, "greedy")
    # 将模型设置为评估模式
    model.eval()
    # 初始化用于存储代价和预测结果的列表
    c = []
    p = []

    def eval_model_bat(bat):
        """
        批量评估模型的辅助函数。

        参数:
        - bat: 当前批次的数据。

        返回:
        - cost: 当前批次的代价数据。
        - pi: 当前批次的预测结果。
        """
        # 禁止梯度计算，因为是在评估模型
        with torch.no_grad():
            # 前向传播，获取代价和预测结果
            cost, _, pi = model(mat, move_to(bat, opts.device))
        # 返回代价和预测结果
        return cost.data.cpu(), pi.data.cpu()

    # 遍历数据集的每个批次
    for bat in DataLoader(dataset, batch_size=opts.eval_batch_size):
        # 对当前批次进行评估，获取代价和预测结果
        cost, pi = eval_model_bat(bat)
        # 遍历当前批次的所有样本
        for z in range(cost.size(0)):
            # 将代价和预测结果添加到列表中
            c.append(cost[z])
            p.append(pi[z])
    # 将列表转换为张量，并返回预测结果和代价
    return torch.stack(p), torch.stack(c)

def set_decode_type(model, decode_type):
    """
    设置模型的解码类型。

    本函数的主要作用是向模型指定使用哪种解码类型，这对于模型处理数据时的解码策略至关重要。

    参数:
    model: 模型对象，具有设置解码类型的方法。
    decode_type: 解码类型，决定了模型如何解码数据。

    返回值:
    无。但通过改变模型的解码类型，影响模型的行为。
    """
    model.set_decode_type(decode_type)

def torch_load_cpu(load_path):
    """
    使用CPU加载PyTorch模型或数据。

    该函数通过指定map_location参数，确保模型或数据即使是在不同类型的设备（如GPU到CPU）上也能正确加载。
    主要用于跨设备迁移学习场景中，确保加载过程的设备兼容性。

    参数:
    - load_path (str): 模型或数据的加载路径。

    返回:
    - 加载的模型或数据对象。
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # 使用lambda函数指定存储位置，确保数据加载到CPU上


def get_inner_model(model):
    """
    返回传入的模型对象本身。

    此函数的目的是提供一个统一的接口，以获取模型的内部表示，无论该模型是否被封装在其他结构中。
    在实际使用中，这可以方便地获取模型的核心部分，以便进行进一步的操作或检查。

    参数:
    model: 任何模型对象 - 该函数的输入参数可以是任何类型的模型对象。

    返回值:
    返回输入的模型对象本身。
    """
    return model


def move_to(var, device):
    """
    将变量或变量字典移动到指定的设备上。

    该函数递归地处理输入变量，如果输入是一个字典，
    它将对字典中的每个值递归调用自身，直到所有的值都被移动到指定的设备上。
    如果输入不是字典，函数会直接将该变量移动到指定的设备上。

    参数:
    var: 要移动的变量或者包含变量的字典。
    device: 目标设备，如 'cpu' 或 'cuda'。

    返回:
    移动到指定设备上的变量或者变量字典。
    """
    # 如果输入是一个字典，则递归地处理字典中的每个值
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    # 否则，直接将变量移动到指定的设备上
    return var.to(device)


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    """
    记录训练过程中的成本和梯度规范值。

    参数:
    - cost: 当前批次的成本张量。
    - grad_norms: 一个元组，包含未裁剪和裁剪后的梯度规范。
    - epoch: 当前的训练周期数。
    - batch_id: 当前的批次ID。
    - step: 全局训练步数。
    - log_likelihood: 对数似然张量。
    - reinforce_loss: REINFORCE损失。
    - bl_loss: 基线损失（例如，批评家损失）。
    - tb_logger: 用于记录到TensorBoard的日志记录器。
    - opts: 包含训练选项的配置对象。

    返回值: 无
    """
    # 计算平均成本并转换为标量
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # 在控制台输出当前训练批次的平均成本
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    # 输出未裁剪和裁剪后的梯度规范
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # 如果配置中启用了TensorBoard，则记录值到TensorBoard
    if not opts.no_tensorboard:
        # 记录平均成本
        tb_logger.log_value('avg_cost', avg_cost, step)

        # 记录REINFORCE损失和负对数似然
        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        # 记录梯度规范和裁剪后的梯度规范
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        # 如果基线方法是‘critic’，记录批评家损失和梯度规范
        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)


class TSPDataset(Dataset):
    
    def __init__(self, ci, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        """
        初始化TSP数据集。

        参数:
        - ci: 城市信息对象，包含城市数量和坐标。
        - filename: 数据集文件名，如果为None，则生成新的数据集。
        - size: 每个样本的序列长度。
        - num_samples: 生成的样本数量。
        - offset: 序列的起始偏移量。
        - distribution: 序列的分布方式，目前未使用。
        """
        # 调用父类的初始化方法
        super(TSPDataset, self).__init__()

        # 如果没有提供文件名，则生成新的数据集
        if (filename is None):
            # 初始化数据集为空列表
            self.data_set = []
            # 生成随机序列，并按升序排序
            l = torch.rand((num_samples, ci.n_cities - 1))
            sorted, ind = torch.sort(l)
            # 扩展索引维度并增加偏移量
            ind = ind.unsqueeze(2).expand(num_samples, ci.n_cities - 1, 2)
            ind = ind[:,:size,:] + 1
            # 复制城市坐标并根据索引获取对应城市序列
            ff = ci.cities.unsqueeze(0)
            ff = ff.expand(num_samples, ci.n_cities, 2)
            f = torch.gather(ff, dim = 1, index = ind)
            f = f.permute(0,2,1)
            # 添加仓库（起始点）到序列
            depot = ci.cities[0].view(1, 2, 1).expand(num_samples, 2, 1)
            self.static = torch.cat((depot, f), dim = 2)
            # 准备索引序列
            depot = torch.zeros(num_samples, 1, 1, dtype=torch.long)
            ind = ind[:,:,0:1]
            ind = torch.cat((depot, ind), dim=1)
        # 如果提供了文件名，则从文件加载数据集
        else:
            # 从文件加载数据并转换为张量
            ff = np.loadtxt(filename, delimiter = ' ')
            ind = torch.tensor(ff, dtype=torch.long).unsqueeze(2)
        # 准备数据集
        self.data = torch.zeros(num_samples, size+1, ci.n_cities)
        self.data = self.data.scatter_(2, ind, 1.)
        # 设置数据集大小
        self.size = len(self.data)

    def __len__(self):
        # 返回对象的大小
        return self.size


    def __getitem__(self, idx):
        """
        通过索引访问数据。

        该方法允许类的实例通过索引访问其内部数据，类似于列表或其他序列类型的行为。

        参数:
        idx (int): 访问数据时的索引位置。

        返回:
        任意类型: 返回位于索引位置 `idx` 的数据项。
        """
        return self.data[idx]




def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    对所有参数组的梯度范数进行裁剪，并返回裁剪前的梯度范数
    :param param_groups: 参数组列表，每个参数组包含一组参数和可能的其他信息
    :param max_norm: float, 指定的用于裁剪的最大范数值，如果为inf，则不进行裁剪
    :return: grad_norms, clipped_grad_norms: 分别为裁剪前和裁剪后的梯度范数列表，每个元素对应一个参数组
    """
    #print(len(param_groups[0]['params']))
    #print('param_groups', param_groups)
    #print('group[params]', [group['params'] for group in param_groups])
    # 初始化梯度范数列表
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],  # 参数组中的参数
            max_norm if max_norm > 0 else math.inf,  # 实际上，如果max_norm大于0，则使用max_norm作为裁剪阈值，否则使用inf不进行裁剪
            norm_type=2  # 使用L2范数
        )
        for group in param_groups  # 对每个参数组计算梯度范数
    ]
    #print(len(param_groups[0]['params']))
    #print('ss', [g_norm for g_norm in grad_norms])
    #print('grad_norms', grad_norms)
    # 根据max_norm的值，裁剪梯度范数
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    #print('grad_norms_clipped', grad_norms_clipped)
    # 返回裁剪前和裁剪后的梯度范数列表
    return grad_norms, grad_norms_clipped


def validate(mat, model, dataset, opts):
    """
    对模型进行验证。

    该函数使用给定的矩阵、模型、数据集和选项来评估模型的性能。
    它通过执行模型的推出过程来计算成本，并报告验证过程的平均成本和标准差。

    参数:
    - mat: 输入矩阵，用于验证过程。
    - model: 需要验证的模型。
    - dataset: 验证数据集。
    - opts: 验证过程的选项，可能包含配置参数等。

    返回:
    - avg_cost: 验证过程的平均成本。
    """
    # 开始验证过程
    print('Validating...')
    # 执行模型推出过程，计算成本
    cost = rollout(mat, model, dataset, opts)
    # 计算平均成本
    avg_cost = cost.mean()
    # 打印验证结果，包括平均成本和标准差
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    # 返回验证的平均成本
    return avg_cost



def train_batch(
        mat,
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    # 从批次数据中分离出输入和基线值
    x, bl_val = baseline.unwrap_batch(batch)
    # 将输入数据移动到指定的设备上
    x = move_to(x, opts.device)
    # 如果基线值存在，也移动到指定设备，否则设置为None
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    # 评估模型，获取成本和对数概率
    cost, log_likelihood, _ = model(mat, x)

    # 评估基线，如果有，获取基线损失（仅对于critic）
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # 计算强化学习损失
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    # 总损失为强化学习损失加上基线损失
    loss = reinforce_loss + bl_loss

    # 清空梯度并进行反向传播和优化步骤
    optimizer.zero_grad()
    loss.backward()
    # 控制梯度规范，进行梯度裁剪，并记录裁剪后的梯度规范
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # 日志记录
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)


def train_epoch(mat, ci, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
    """
    训练模型一个epoch。

    参数:
    mat: 训练过程中的矩阵数据。
    ci: 问题实例的成本矩阵。
    model: 要训练的模型。
    optimizer: 优化器，用于更新模型参数。
    baseline: 基线模型，用于加速训练。
    lr_scheduler: 学习率调度器，用于调整学习率。
    epoch: 当前的训练周期数。
    val_dataset: 验证数据集，用于评估模型性能。
    tb_logger: TensorBoard日志记录器，用于记录训练过程中的数据。
    opts: 用户指定的训练选项，如运行名称、图大小等。

    返回:
    无。
    """
    # 打印开始训练的提示信息，包括当前epoch、学习率和运行名称
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    # 计算当前的训练步数
    step = epoch * (opts.epoch_size // opts.batch_size)
    # 记录开始时间
    start_time = time.time()
    # 调度学习率
    lr_scheduler.step(epoch)

    # 如果启用了TensorBoard，记录当前学习率
    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # 为每个epoch生成新的训练数据
    training_dataset = baseline.wrap_dataset(TSPDataset(ci, size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

    # 将模型设置为训练模式，并设置解码类型为采样
    model.train()
    set_decode_type(model, "sampling")

    # 遍历训练数据集的每个批次进行训练
    for batch_id, batch in enumerate(training_dataloader):
        # 执行一个训练步骤
        train_batch(
            mat,
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        # 更新步数
        step += 1

    # 计算并打印一个epoch的训练所需时间
    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # 如果达到指定的checkpoint周期或最后一个epoch，保存模型和相关状态
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    # 在验证集上评估模型性能
    avg_reward = validate(mat, model, val_dataset, opts)

    # 如果启用了TensorBoard，记录验证集上的平均回报
    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    # 调用基线模型的回调函数
    baseline.epoch_callback(model, epoch)




def run(opts):
    # 打印运行参数
    print(123)
    pp.pprint(vars(opts))

    # 设置随机种子
    torch.manual_seed(opts.seed)

    # 配置Tensorboard日志记录器
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    # 创建保存目录
    os.makedirs(opts.save_dir)
    # 保存运行参数，以便将来查找
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # 设置设备（CUDA或CPU）
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    ci = Cities()
    mat = DistanceMatrix(ci, load_dir='./m1/data.csv', max_time_step = 12)
    # 保存矩阵数据
    np.savetxt('mat.txt', mat.mat.cpu().numpy(), fmt='%.6f')
    np.savetxt('m2.txt', mat.m2.cpu().numpy(), fmt='%.6f')
    np.savetxt('m3.txt', mat.m3.cpu().numpy(), fmt='%.6f')
    np.savetxt('m4.txt', mat.m4.cpu().numpy(), fmt='%.6f')

    # 初始化模型
    model_class = AttentionModel
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        input_size=opts.graph_size+1,
        max_t=12
    ).to(opts.device)

    # 加载模型参数
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # 初始化基线
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(mat, ci, model, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # 渐热基线
    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # 加载基线状态
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # 初始化优化器
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # 加载优化器状态
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # 初始化学习率调度器
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    # 开始实际的训练循环
    val_dataset = TSPDataset(ci, size=opts.graph_size, num_samples=opts.val_size, filename='data_nodes/node_19.txt', distribution=opts.data_distribution)
    _, ind = torch.max(val_dataset.data, dim=2)
    # 恢复训练
    if opts.resume:
        epoch_resume = 999
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # 使用基线模型进行滚动预测
    model2 = baseline.baseline.model
    ans, cost = roll(mat, model2, val_dataset, opts)
    print('Avg cost:', torch.mean(cost) * 1440)
    np.savetxt('answer.txt', ans.numpy(), fmt='%d')
    np.savetxt('costs.txt', cost.numpy(), fmt='%.6f')
if __name__ == "__main__":
    run(get_options())
