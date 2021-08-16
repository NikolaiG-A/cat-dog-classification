import os
import torch
from torch.nn.functional import softmax
import torchvision.transforms as T
from PIL import Image
from io import BytesIO

from models.custom import ImageClassifier
from utils import get_model_path,get_config_yaml

def predict(config_dir,image_bytes,is_cpu = True):
    """
    Function predict applies the model to the image and saves returns results.
    :param config_dir: str - path to directory with a training config file and model weights
    :param image_bytes: bytes - image to run predictions
    :param is_cpu: bool - to run model on a cpu, otherwise on a gpu (if available)

    output:
    result: dict - with fields "name": class_name (str), "confidence": prediction_score (float)
    """
    conf_path = os.path.abspath(config_dir)
    ### get config parameters
    conf = get_config_yaml(conf_path)
    ### class names
    label_names = conf['data']['class_names']
    ### get the last model
    model_path = get_model_path(conf_path)
    
    ### get the device
    device = torch.device("cuda" if torch.cuda.is_available() and not is_cpu else "cpu")
    ### get the model
    model = ImageClassifier(conf)
    ### load the weights
    trainied_model_weights = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(trainied_model_weights["model_state_dict"])
    model.to(device)
    model.eval()
    
    """
    process an image
    """
    transforms = [T.ToTensor()]
    for key in conf['training']['transforms']:
        transform_i = getattr(T, key)
        transforms.append(transform_i(**conf['training']['transforms'][key]))
    transforms = T.Compose(transforms)
    ### open an image
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    ### apply the augmentations
    processed_image = transforms(img)
    ### add one dimension
    processed_image = processed_image.unsqueeze(0)

    ### apply the model
    logits = softmax(model(processed_image),dim=1)
    ### get the label name
    label = torch.argmax(logits).item()
    ### get the probability
    p = round(torch.max(logits).item(),3)
    return {'name':label_names[label],'confidence':p}

if __name__=='__main__':
    ### set up the image path
    img_path = 'dataset/dog/test/dog.12.jpg'
    img_path = os.path.abspath(img_path)
    ### set up the folder with model weights and config
    config_dir = 'output'
    config_dir = os.path.abspath(config_dir)
    
    ### make the prediction
    image_bytes = open(img_path,'rb').read()
    result = predict(config_dir,image_bytes)
    print(result)
