import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from utils.dataset import HSIDataset
from torch.utils.data import DataLoader
from models.resnet import ResNet18_ as Net
from utils.model_wrapper import ModelWrapper, APCER, BPCER, BinaryAUC


def build_dataset(batch_size=16, image_size=128, crop_scale=(0.5, 0.8)):
    trainset = HSIDataset(
        images_dir='./data/HySpeFAS/images',
        label_path='./data/HySpeFAS/train.txt',
        image_size=image_size,
        crop_scale=crop_scale,
        trainset=True,
        positive_rep_rate=3,
    )
    valset = HSIDataset(
        images_dir='./data/HySpeFAS/images',
        label_path='./data/HySpeFAS/val.txt',
        image_size=image_size,
        crop_scale=crop_scale,
        trainset=False,
    )

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train(Model=Net, name='hyspefas-resnet', verbose=1, batch_size=60, do_validation=True, epochs=25,
          **model_kwargs):
    batch_size = batch_size
    epochs = epochs
    model_name = name

    trainloaer, valloader = build_dataset(batch_size=batch_size)
    if not do_validation:
        valloader = None
    model = Model(**model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.MultiStepLR(optim, [10, 15, 20], gamma=0.1)

    loss_fcn = torch.nn.functional.binary_cross_entropy
    apcer = APCER(device=torch.device('cpu'))
    bpcer = BPCER(device=torch.device('cpu'))
    auc = BinaryAUC(device=torch.device('cpu'))

    trainer = ModelWrapper(model, device)
    trainer.compile(
        optimizer=optim,
        loss_fcn=loss_fcn,
        metrics=[apcer, bpcer, auc],
        schedule=schedule,
    )

    trainer.train(
        train_data=trainloaer,
        val_data=valloader,
        epochs=epochs,
        save_path='./checkpoints/{:s}.pth'.format(model_name),
        verbose=verbose,
    )
    rtv = trainer.evaluate(valloader, verbose=verbose)

    torch.save(model.state_dict(), './checkpoints/{:s}.pth'.format(model_name))

    return rtv


if __name__ == '__main__':
    train()