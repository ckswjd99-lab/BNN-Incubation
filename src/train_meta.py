import torch
import torch.nn as nn

import time

from models import ResNetMeta
from dataloader import get_imnet1k_dataloader
from trainer import train, validate

BATCH_SIZE = 128
EPOCHS = 200
LR_START = 1e-3
LR_END = 1e-5
CKPT_NAME = "resnet50_meta.pth"


train_loader, val_loader = get_imnet1k_dataloader(batch_size=BATCH_SIZE, augmentation=False)

model = ResNetMeta(input_size=224, num_output=1000, input_channel=3).cuda()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"Number of parameters: {num_params:,d}")

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_START, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(LR_END/LR_START)**(1/(EPOCHS-1)))

best_vacc = 0
start_epoch = 0

for epoch in range(start_epoch, EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion, None)
    val_loss, val_acc = validate(epoch, model, val_loader, criterion, None)
    scheduler.step()

    is_best = val_acc > best_vacc
    best_vacc = max(val_acc, best_vacc)

    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_vacc': best_vacc
    }
    torch.save(checkpoint, f"saves/{CKPT_NAME}")

    if is_best:
        torch.save(checkpoint, f"saves/{CKPT_NAME.replace('.pth', '_best.pth')}")

    end_time = time.time()
    print(f"EPOCH {epoch:3d}, LR: {scheduler.get_last_lr()[0]:.4e} | T LOSS: {train_loss:.4f}, T ACC: {train_acc*100:.2f}%, V LOSS: {val_loss:.4f}, V ACC: {val_acc*100:.2f}% | ETA: {int(end_time-start_time) // 60:8,d} min")