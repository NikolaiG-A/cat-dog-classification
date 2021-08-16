import torch
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

@torch.no_grad()
def evaluate(model, loss_func, val_loader):
    model.eval()
    outputs = []
    for batch in tqdm(val_loader):
        outputs.append(model.validation_step(batch,loss_func))
    return model.validation_epoch_end(outputs)

def train_one_epoch(model, loss_func, optimizer, lr_scheduler, train_dl,test_dl, epoch):
    ### training phase
    model.train()
    ### keep training losses
    train_losses = []
    ### keep learning rate
    lrs = []
    print('-----Training phase-----')
    for batch in tqdm(train_dl):
        ### set grads to zero
        optimizer.zero_grad()
        ### calculate loss
        loss = model.training_step(batch,loss_func)
        ### accumulate losses
        train_losses.append(loss)
        ### backprop
        loss.backward()
        ### optimizer step
        optimizer.step()        
        ### schedule step
        lrs.append(get_lr(optimizer))
    if lr_scheduler is not None:
        lr_scheduler.step()
    ### validation phase
    print('-----Validation phase-----')
    result = evaluate(model, loss_func, test_dl)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['last_lr'] = lrs[-1]
    result_txt = model.epoch_end(epoch,result)
    return result,result_txt

