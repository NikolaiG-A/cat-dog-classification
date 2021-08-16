import torch
import torch.nn as nn
@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel,self).__init__()
    def training_step(self, batch,loss_func):
        images, labels = batch
        ### make prediction
        out = self(images)
        ### calculate loss
        loss = loss_func(out, labels)
        return loss
    @torch.no_grad()
    def validation_step(self, batch, loss_func):
        images, labels = batch
        ### make prediction
        out = self(images)
        ### calculate loss
        loss = loss_func(out, labels)
        ### calculate accuracy
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        ### Combine losses
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        ### Combine accuracies
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    def epoch_end(self, epoch, result):
        result_txt = "epoch {}, last_lr: {:.2e}, train_loss: {:.3e}, val_loss: {:.3e}, val_acc: {:.4f}".format(
            epoch, result['last_lr'], result['train_loss'], result['val_loss'], result['val_acc'])
        print(result_txt)
        return result_txt