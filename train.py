from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from datamodules import NuplanDataModule
from model import PlanR1
from utils import load_config

import os

CURRENT_FILE_PATH = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train/pred.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    # Load the model from a checkpoint if provided
    if config['trainer']['ckpt_path']:
        print(f"Loading model from checkpoint: {config['trainer']['ckpt_path']}")
        model = PlanR1.load_from_checkpoint(config['trainer']['ckpt_path'], **config['model'])
    else:
        print("No checkpoint path provided, initializing a new model.")
        model = PlanR1(**config['model'])

    # 定义数据模块
    datamodule = NuplanDataModule(**config['datamodule'])
    # 定义训练模块
    model_checkpoint = ModelCheckpoint(**config['trainer']['ckpt'])     # 模型**保存**回调函数，在Plan-R1中保存验证集上reward最优的模型权重
    lr_monitor = LearningRateMonitor(**config['trainer']['lr_monitor']) # 学习率监控回调函数
    csv_logger = CSVLogger(**config['trainer']['csv_logger'])           # 日志记录器
    trainer = pl.Trainer(
        strategy=config['trainer']['strategy'],             # 分布式训练策略
        devices=config['trainer']['devices'],               # 使用的设备数量
        accelerator=config['trainer']['accelerator'],       # 使用的加速器类型
        callbacks=[model_checkpoint, lr_monitor],           # 回调函数列表，包括模型保存和学习率监控
        max_epochs=config['trainer']['max_epochs'],         # 最大训练轮数
        logger=csv_logger                                   # 日志记录器
    )

    trainer.fit(model, datamodule)