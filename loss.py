import torch
import torch.nn as nn

### define a custom loss function
class CustomLoss(nn.Module):
    """
    need to define __init__ and forward methods
    """
    def __init__(self):
        self(CustomLoss,self).__init__()
    def forward(self,pred,labels):
        pass

### define a loss function
def get_loss_function(conf):
    ### get loss function from the config
    if conf['training']['loss']['name'] == 'custom':
        loss = CustomLoss()
    else:
        loss_func = getattr(nn, conf['training']['loss']['name'])
        loss_param = conf['training']['loss'].copy()
        del loss_param['name']
        for key in loss_param:
            if key in ['weight','pos_weight']:
                loss_param[key] = torch.tensor(loss_param[key])
        loss = loss_func(**loss_param)
    return loss