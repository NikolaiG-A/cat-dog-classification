import torch
import yaml
import os

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
def load_yaml(config_path):
    """
    Function load_yaml loads a yaml file from its path
    :param config_path: str - path to a .yaml file with training parameters

    output:
    config_path: str - (dict) configuration parameters
    """
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config
def get_config_yaml(config_dir):
    """
    Function get_config_yaml returns the path to a .yaml file with training parameters from its directory
    :param config_dir: str - path to directory with a training config file and model weights

    output:
    config_path: str - config path
    """
    for file_i in os.listdir(config_dir):
        if '.yaml' in file_i:
            conf_file=file_i
            break
    config_path = load_yaml(os.path.join(config_dir,conf_file))
    return config_path
def get_model_path(config_dir):
    """
    Function get_model_path returns the model path from its directory
    :param config_dir: str - path to directory with a training config file and model weights

    output:
    model_path: str - model path
    """
    ### get the model with the largest number in the name
    model_files = sorted([model_i for model_i in os.listdir(config_dir) if 'model' in model_i])
    if len(model_files)>1:
        model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    ### get the last model
    model_path = os.path.join(config_dir,model_files[-1])
    return model_path
def save_tb(tb_logger,model, result, epoch):
    ### scalar values
    for key in result:
        tb_logger.add_scalar(key, result[key], epoch)

    ### histograms of weights and their gradients
    for tag, value in model.named_parameters():
        tag = tag.replace(".", "-")
        tb_logger.add_histogram(tag, value.data.cpu().numpy(), epoch)
        if not value.grad is None:
            tb_logger.add_histogram(tag + "-grad", value.grad.data.cpu().numpy(), epoch)
    pass