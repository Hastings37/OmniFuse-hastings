import torch
from torch.utils.data import DataLoader
import yaml
import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (即 complex_degradation 的上一级目录)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from complex_degradation.Complex_degradation_dataset import ControlFusionDataset, DiffusionFusionDataset_train, \
    DiffusionFusionDataset_test


# 给定需要加载的yaml 文件的路径位置；
def load_yaml_config(file_path):
    """
    通用函数：将 YAML 文件转换为 Python 字典

    Args:
        file_path (str): YAML 文件的路径

    Returns:
        dict: 转换后的字典
        None: 如果读取失败
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: 文件不存在 - {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # safe_load 是推荐的安全加载方式，防止执行恶意代码
            config_dict = yaml.safe_load(file)
            print(f"Success: 成功加载配置文件 - {file_path}")
            return config_dict

    except yaml.YAMLError as exc:
        print(f"Error: YAML 格式解析错误 - {exc}")
        return None
    except Exception as e:
        print(f"Error: 读取文件时发生未知错误 - {e}")
        return None


# ================= 使用示例 =================

def create_dataloader(dataset, phase, opt):
    """
    通用 DataLoader 创建函数
    """
    if dataset is None:
        return None

    batch_size = opt.get('batch_size_per_gpu', 1)
    num_workers = opt.get('num_workers_per_gpu', 0)

    # 训练集打乱，验证/测试集不打乱
    shuffle = True if phase == 'train' else False

    # 动态获取 dataset 的 collate_fn，适配不同的 Dataset 类
    collate_fn = getattr(dataset, 'collate_fn', None)
    if collate_fn is None:
        raise ValueError("Dataset must have a static method 'collate_fn'")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    print(f"[{phase}] DataLoader 创建成功: Batch Size={batch_size}, Num Workers={num_workers}, Shuffle={shuffle}")
    return dataloader


def get_controlfusion_dataloaders():
    # ================= 配置路径 =================
    # 请根据实际情况修改 YAML 文件的路径
    yaml_path = "data/train_diffusion/train_diffusion.yaml"
    # 如果你在本地测试，可能需要临时改为:
    # yaml_path = "dataset_config.yaml"

    print(f"正在读取配置文件: {yaml_path}")

    # 1. 加载配置
    config = load_yaml_config(yaml_path)

    if config:
        # 处理 YAML 结构的层级
        # 有些 YAML 根节点就是 dataset，有些根节点包含 dataset 键
        # 这里做一个兼容处理：如果根节点下有 'dataset'，就取进去；否则直接用根节点
        if 'dataset' in config:
            dataset_config = config['dataset']
        else:
            dataset_config = config

        train_set, val_set = None, None
        train_loader, val_loader = None, None

        # 2. 遍历配置构建 Dataset 和 DataLoader
        for key, value in dataset_config.items():
            if key == 'train':
                print("\n=== 正在构建训练集 (Train Dataset) ===")
                try:
                    # 实例化 Dataset
                    train_set = ControlFusionDataset(value)
                    print(f"训练集样本数量: {len(train_set)}")

                    # 构建 DataLoader
                    train_loader = create_dataloader(train_set, 'train', value)
                except Exception as e:
                    print(f"构建训练集失败: {e}")
                    import traceback

                    traceback.print_exc()

            elif key == 'val':
                print("\n=== 正在构建验证集 (Val Dataset) ===")
                try:
                    # 实例化 Dataset
                    val_set = ControlFusionDataset(value)
                    print(f"验证集样本数量: {len(val_set)}")

                    # 构建 DataLoader
                    val_loader = create_dataloader(val_set, 'val', value)
                except Exception as e:
                    print(f"构建验证集失败: {e}")
                    import traceback

                    traceback.print_exc()

        # 3. 简单的测试循环 (Sanity Check)
        print("\n=== 开始 DataLoader 测试 (仅取一个 Batch) ===")

        # 测试训练集
        if train_loader:
            print("\n>>> Testing Train Loader:")
            try:
                for i, batch in enumerate(train_loader):
                    # 解包 batch，对应 collate_fn 的返回值
                    # vis, ir, vis_gt, ir_gt, full, text, name = batch
                    vis, ir, vis_gt, ir_gt, full, text, name = batch

                    print(f"Batch Index: {i}")
                    print(f"  Visible Shape:   {vis.shape}")
                    print(f"  Infrared Shape:  {ir.shape}")
                    print(f"  Vis GT Shape:    {vis_gt.shape}")
                    print(f"  IR GT Shape:     {ir_gt.shape}")
                    print(f"  Full Shape:      {full.shape}")
                    print(f"  Text Prompt[0]:  {text[0]}")  # 打印第一个样本的文本提示
                    print(f"  Filename[0]:     {name[0]}")  # 打印第一个样本的文件名

                    # 仅测试一个 batch 就退出
                    break
                print("Train Loader 测试通过 ✅")
            except Exception as e:
                print(f"Train Loader 测试失败 ❌: {e}")
                import traceback

                traceback.print_exc()

        # 测试验证集
        if val_loader:
            print("\n>>> Testing Val Loader:")
            try:
                for i, batch in enumerate(val_loader):
                    # 注意：根据你的 Dataset 代码注释，val 返回的内容可能不一样
                    # 但 collate_fn 统一处理成了 7 个返回值，如果 val __getitem__ 返回值数量不同，这里可能会报错
                    # 让我们查看你的 __getitem__ 代码...
                    # 看起来你的 __getitem__ 不区分 train/val 返回同样的 7 个元素 (vis, ir, vis_gt, ir_gt, full, text, name)
                    # 所以这里解包应该是安全的

                    vis, ir, vis_gt, ir_gt, full, text, name = batch

                    print(f"Batch Index: {i}")
                    print(f"  Visible Shape:   {vis.shape}")
                    print(f"  Infrared Shape:  {ir.shape}")
                    print(f"  Vis GT Shape:    {vis_gt.shape}")
                    print(f"  IR GT Shape:     {ir_gt.shape}")

                    print(f"  Text Prompt[0]:  {text[0]}")
                    print(f"  Filename[0]:     {name[0]}")

                    break
                print("Val Loader 测试通过 ✅")
            except Exception as e:
                print(f"Val Loader 测试失败 ❌: {e}")
                import traceback

                traceback.print_exc()

    else:
        print("未加载到有效配置，程序结束。")

    return train_loader, val_loader


def get_my_train_dataloaders(config):
    if not config:
        print("未加载到有效配置。")
        return None, None

    # 兼容处理 dataset 节点
    dataset_config = config.get('dataset', config)

    train_loader, val_loader = None, None

    # 1. 构建 Train Loader
    if 'train' in dataset_config:
        print("\n=== 正在构建训练集 (DiffusionFusionDataset_train) ===")
        try:
            value = dataset_config['train']
            # 注意：这里传入 MAX_SIZE 和 CROP_SIZE
            CROP_SIZE = dataset_config['train'].get('crop_size', 512)
            MAX_SIZE = dataset_config['train'].get('max_size', 640)
            train_set = DiffusionFusionDataset_train(MAX_SIZE, CROP_SIZE, value)
            print(f"训练集样本数量: {len(train_set)}")
            train_loader = create_dataloader(train_set, 'train', value)
        except Exception as e:
            print(f"构建训练集失败: {e}")
            import traceback
            traceback.print_exc()

    # 2. 构建 Val Loader (通常在训练yaml中也有val部分用于监控)
    # 这里的策略是：训练时的验证集应该使用 Test Dataset 的逻辑（不做随机裁剪），以便指标稳定
    if 'val' in dataset_config:
        print("\n=== 正在构建验证集 (DiffusionFusionDataset_test) ===")
        try:
            value = dataset_config['val']
            # 验证集通常不需要 CROP_SIZE，只需要 MAX_SIZE 做对齐
            MAX_SIZE = dataset_config['val'].get('max_size', 640)
            CROP_SIZE = dataset_config['val'].get('crop_size', 512)
            val_set = DiffusionFusionDataset_train(MAX_SIZE, CROP_SIZE, value)  # 使用的也是test下的对应内容；
            print(f"验证集样本数量: {len(val_set)}")
            val_loader = create_dataloader(val_set, 'val', value)  # 接受剩余的参数内容；
        except Exception as e:
            print(f"构建验证集失败: {e}")
            import traceback
            traceback.print_exc()

    # 3. 简单的 Sanity Check (解包 11 个变量)
    if train_loader:
        print("\n>>> Testing Train Loader (First Batch):")
        try:
            for i, batch in enumerate(train_loader):
                # 解包 11 个返回值
                vis, vis_gt, ir, ir_gt, type_name, text, vis_name, ir_name = batch
                print(f'vis.range: {vis.min().item()} ~ {vis.max().item()}')

                print(f"  Vis Y Shape:     {vis.shape}")  # [B, 1, H, W]
                print(f"  Vis GT Y Shape:  {vis_gt.shape}")
                print(f"  IR Y Shape:      {ir.shape}")
                print(f'  IR GT Y Shape:   {ir_gt.shape}')
                print(f'type_name: {type_name}')
                # 这里的形式都是 Y 分量的内容；
                print(f"  Prompt[0]:       {text[0]}")
                print(f"  Vis Name[0]:     {vis_name[0]}")
                break
            print("Train Loader 测试通过 ✅")
        except Exception as e:
            print(f"Train Loader 测试失败 ❌: {e}")
            import traceback
            traceback.print_exc()

    if val_loader:
        print("\n>>> Testing Val Loader (First Batch):")
        try:
            for i, batch in enumerate(val_loader):
                vis, vis_gt, ir, ir_gt, type_name, text, vis_name, ir_name = batch
                print(f'vis.range: {vis.min().item()} ~ {vis.max().item()}')

                print(f"  Vis Y Shape:     {vis.shape}")  # [B, 1, H, W]
                print(f"  Vis GT Y Shape:  {vis_gt.shape}")
                print(f"  IR Y Shape:      {ir.shape}")
                print(f'  IR GT Y Shape:   {ir_gt.shape}')
                print(f'type_name: {type_name}')
                # 这里的形式都是 Y 分量的内容；
                print(f"  Prompt[0]:       {text[0]}")
                print(f"  Vis Name[0]:     {vis_name[0]}")
                break
            print("Val Loader 测试通过 ✅")
        except Exception as e:
            print(f"Val Loader 测试失败 ❌: {e}")
            import traceback
            traceback.print_exc()

    return train_loader, val_loader


def get_my_test_dataloaders(config):
    """
    专门用于获取测试/推理流程的 Loader
    仅使用 DiffusionFusionDataset_test
    """
    # print(f"正在读取测试配置文件: {yaml_path}")

    # config = load_yaml_config(yaml_path)
    if not config:
        return None

    dataset_config = config.get('dataset', config)
    test_loader = None

    # 测试配置中通常 key 可能是 'test' 或者 'val'，这里遍历查找
    # 假设测试集配置在 'val' 或者 'test' 字段下
    # 这里给定的本来就是config中的 dataset 项目的内容了；
    target_keys = ['test']

    for key in target_keys:
        if key in dataset_config:
            print(f"\n=== 正在构建测试集 (Key: {key}) ===")
            try:
                value = dataset_config[key]
                # Test Dataset 只需要 MAX_SIZE
                MAX_SIZE = dataset_config[key].get('max_size', 512)
                # 应该就是

                test_set = DiffusionFusionDataset_test(MAX_SIZE, value)
                print(f"测试集样本数量: {len(test_set)}")

                # phase 设为 'test' 以便不打乱
                test_loader = create_dataloader(test_set, 'test', value)

                # 找到一个就退出循环 (或者你可以根据需求返回多个)
                break
            except Exception as e:
                print(f"构建测试集失败: {e}")
                import traceback
                traceback.print_exc()

    # Sanity Check
    if test_loader:
        print("\n>>> Testing Test Loader (First Batch):")
        try:
            for i, batch in enumerate(test_loader):
                vis, vis_gt, ir, ir_gt, type_name, text, vis_name, ir_name = batch

                print(f"  Vis Y Shape:     {vis.shape}")  # [B, 1, H, W]
                print(f"  Vis GT Y Shape:  {vis_gt.shape}")
                print(f"  IR Y Shape:      {ir.shape}")
                print(f'  IR GT Y Shape:   {ir_gt.shape}')
                # 这里的形式都是 Y 分量的内容；
                print(f'type_name: {type_name}')
                print(f"  Prompt[0]:       {text[0]}")
                print(f"  Vis Name[0]:     {vis_name[0]}")
                break
            print("Test Loader 测试通过 ✅")
        except Exception as e:
            print(f"Test Loader 测试失败 ❌: {e}")
            import traceback
            traceback.print_exc()

    return test_loader


# ==========================================
# 使用示例 (可以放在 if __name__ == "__main__": 中运行)
# ==========================================
if __name__ == "__main__":
    # 假设你的 yaml 路径如下
    train_yaml = "option/train/train_diffusion.yaml"

    opt = load_yaml_config(train_yaml)
    config = opt['dataset']

    # 1. 获取训练用的 Loader
    # train_dl 会有随机裁剪 (CROP_SIZE=128)
    # val_dl (如有) 会是全图 Resize (MAX_SIZE=2048)
    train_loader, val_loader = get_my_train_dataloaders(
        config
    )

    # 2. 获取测试用的 Loader
    # test_dl 仅做 Resize

    # test_loader = get_my_test_dataloaders(config)
