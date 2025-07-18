from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV1Dataset
from models.ise import ISE

if __name__ == '__main__':
    pl.seed_everything(2024)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=False, default='argoverse')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=False, default='ckpt')
    args = parser.parse_args()

    trainer = pl.Trainer()
    model = ISE.load_from_checkpoint(checkpoint_path=args.ckpt_path, parallel=False)
    val_dataset = ArgoverseV1Dataset(root=args.root, split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer.validate(model, dataloader)
