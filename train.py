from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch_geometric.loader import DataLoader

from datasets import  ArgoverseV1Dataset
# from datamodules import ArgoverseV1DataModule
from models.ise import ISE
import torch
from torch.utils.data import Subset
import numpy as np

if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=False, default='/path/argoverse/')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=15)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=30)
    parser = ISE.add_model_specific_args(parser)
    args = parser.parse_args()

    logger = TensorBoardLogger(save_dir='tf-logs', name="logs")
    model_checkpoint = ModelCheckpoint(dirpath='model_checkpoint/',filename='model-{epoch:02d}',monitor=args.monitor, save_top_k=args.save_top_k, mode='min')    
    trainer = pl.Trainer(max_epochs=128,accelerator='gpu', log_every_n_steps=50,callbacks=[model_checkpoint],
                         logger=logger)
    model = ISE(**vars(args))

    train_dataset = ArgoverseV1Dataset(root=args.root, split='train')
    train_dataloader = DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,persistent_workers=True)
    val_dataset = ArgoverseV1Dataset(root=args.root, split='train')
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True, persistent_workers=True)

    torch.set_float32_matmul_precision('high')
    trainer.fit(model,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
