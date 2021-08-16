import torch
import os
import shutil
import torchvision.transforms as T
import tensorboardX

from models.custom import ImageClassifier
from engine import train_one_epoch
from datasets import ImagesDataset
import utils
from loss import get_loss_function

config_path = os.path.abspath('./config.yaml')
conf = utils.load_yaml(config_path)

def main():
    ### train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() and not conf['device'] == 'cpu' else torch.device('cpu')
    
    """
    set up transofrms
    """
    transforms = [T.ToTensor()]
    for key in conf['training']['transforms']:
        transform_i = getattr(T, key)
        transforms.append(transform_i(**conf['training']['transforms'][key]))
    transforms = T.Compose(transforms)

    """
    define datasets and dataloaders (move to device)
    """
    train_data = ImagesDataset(conf, is_train = True, transforms = transforms)
    test_data = ImagesDataset(conf, is_train = False, transforms = transforms)

    ### define training and validation data loaders
    train_dl = torch.utils.data.DataLoader(
                                            train_data,
                                            batch_size=conf['training']['batch_size'], 
                                            shuffle=True,
                                            num_workers=conf['training']['num_workers'],
                                            pin_memory = conf['training']['pin_memory']
                                           )
    test_dl = torch.utils.data.DataLoader(
                                            test_data,
                                            batch_size=conf['training']['batch_size'], 
                                            shuffle=False,
                                            num_workers=conf['training']['num_workers'],
                                            pin_memory = conf['training']['pin_memory']
                                           )
    ### move loaders to the corresponding device
    train_dl = utils.DeviceDataLoader(train_dl,device)
    test_dl = utils.DeviceDataLoader(test_dl,device)
    """
    prepare the model and optimizers
    """
    ### get the model using our helper function
    model = ImageClassifier(conf)

    ### parameters to modify
    params = [p for p in model.parameters() if p.requires_grad]

    ### construct an optimizer
    opt_func = getattr(torch.optim, conf['training']['optimizer']['name'])
    
    opt_param = conf['training']['optimizer'].copy()
    del opt_param['name']
    optimizer = opt_func(params, **opt_param)

    # and a learning rate scheduler
    if conf['training']['lr_schedule']:
        lr_scheduler_func = getattr(torch.optim.lr_scheduler, conf['training']['lr_schedule']['name'])
        lr_schedule_param = conf['training']['lr_schedule'].copy()
        del lr_schedule_param['name']
        lr_scheduler = lr_scheduler_func(optimizer,**lr_schedule_param)
    else:
        lr_scheduler = None
    # get the number of epochs
    num_epochs = conf['training']['nb_epochs']
    # starting epoch
    start_epoch = 0
    """
    get loss function
    """
    loss = get_loss_function(conf)
    """
    set up the saving 
    """
    ### saving path
    saving_path = os.path.abspath(conf['saving']['output_folder'])
    ### delete the output directory if exists
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)
        shutil.copy(config_path,os.path.join(saving_path,'config.yaml'))
    else:
        if conf['training']['resume'] and not conf['saving']['save']=='best':
            ### get the last model
            try:
                model_path = utils.get_model_path(saving_path)
                ### load the weights and optimizer state
                model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
                model.load_state_dict(model_ckpt["model_state_dict"])
                optimizer.load_state_dict(model_ckpt['optimizer_state_dict'])
                if conf['training']['lr_schedule']:
                    lr_scheduler.load_state_dict(model_ckpt['lr_scheduler_state_dict'])
                shutil.copy(config_path,os.path.join(saving_path,'config_'+str(model_ckpt["epoch"])+'.yaml'))
                ### change the starting epoch
                start_epoch = model_ckpt["epoch"]-1
            except ValueError:
                print("If 'resume' option is true, checkpoint files must preset in output folder.\n Otherwise clean this directory")
        else:
            for files in os.listdir(saving_path):
                path = os.path.join(saving_path, files)
                try:
                    shutil.rmtree(path)
                except OSError:
                    os.remove(path)
            shutil.copy(config_path,os.path.join(saving_path,'config.yaml'))
    ### move model to the right device
    model.to(device)

    ### try to maximize the accuracy
    save_score = 0
    ### output file to keep the history
    out_file = open(os.path.join(saving_path,'history.txt'), 'w')
    ### set up tensorboard
    tb_log = tensorboardX.SummaryWriter(log_dir=saving_path)

    """
    run the training
    """
    for epoch in range(start_epoch,num_epochs):
        ### train for one epoch
        result,result_txt = train_one_epoch(model, loss, optimizer, lr_scheduler, train_dl,test_dl, epoch+1)
        out_file.write(result_txt+'\n')
        ### save tensorboard
        utils.save_tb(tb_log,model,result,epoch+1)
        ### checkpoint results
        model_ckpt = {
                        "epoch": epoch+1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()
                     }
        if conf['training']['lr_schedule']:
            model_ckpt["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        ### save model results
        if conf['saving']['save'] == 'best':
            if epoch==0:
                torch.save(model_ckpt, os.path.join(saving_path,'model_best.ckpt'))
            else:
                if result['val_acc']>save_score:
                    torch.save(model_ckpt, os.path.join(saving_path,'model_best.ckpt'))
                    save_score=result['val_acc']
        if conf['saving']['save'] == 'all':
            torch.save(model_ckpt,os.path.join(saving_path,'model_{0}.ckpt'.format(epoch+1)))
    ### close the history file
    out_file.close()
if __name__=='__main__':
    main()