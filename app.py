# -*- coding: utf-8 -*-
import os
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename

import config
from predict import predict
from utils import get_config_yaml

### define model path and load config parameters
model_path = os.path.abspath(config.MODEL_PATH)
confs  = get_config_yaml(model_path)
categories = confs['data']['class_names']


app = Flask(__name__,template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__),'templates')))

app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__),'img_download'))
if not os.path.isdir(app.config['UPLOAD_FOLDER']):
    os.mkdir(app.config['UPLOAD_FOLDER'])

### make folders for correct and wrong predictions
true_folder = os.path.join(app.config['UPLOAD_FOLDER'],'true')
false_folder = os.path.join(app.config['UPLOAD_FOLDER'],'false')
for fold_i in [true_folder,false_folder]:
    if not os.path.isdir(fold_i):
        os.mkdir(fold_i)
    for cat_i in categories:
        if not os.path.isdir(os.path.join(fold_i,cat_i)):
            os.mkdir(os.path.join(fold_i,cat_i))

### define global parameters which will be changed after predictions
result = {'name':'cat','confidence':0.0}
file_save_path = 'example.jpg'

@app.route('/', methods = ['GET','POST'])
def predict_image():
    """
    Function predict_image applies the model to the image and saves results.
    The global parameters are updated:
    result: dict - with fields "name": class_name (str), "confidence": prediction_score (float)
    file_save_path: str - temprorary path to save image
    """
    global file_save_path,result
    try:
        if request.method == 'POST':
            img_file = request.files['file']
            predict_result = predict(model_path,img_file.read())
            filename = secure_filename(img_file.filename)
            file_save_path_call = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            ### update values of global parameters
            result['name'] = predict_result['name']
            result['confidence'] = predict_result['confidence']
            file_save_path = file_save_path_call
            img_file.save(file_save_path_call)
            return redirect(url_for('check_category'))
        return render_template(config.UPLOAD_FILE_HTML)
    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500


@app.route('/check_category', methods = ['GET','POST'])
def check_category():
    """
    Function check_category allows to confirm the prediction.
    The 'Skip' button will remove the loaded image file.
    The 'Submit' button will move the loaded file to 'true' or 'false' prediction according to user's confirmation.
    Folders 'true' and 'false' contain the subdirectories with all class names 
    """
    if request.method == 'POST':
        if request.form['submit_button'] == 'Delete':
            os.remove(file_save_path)
        else:
            select = request.form.get('comp_select')
            if result['name'] == select:
                fold_save = os.path.join(true_folder,select)
            else:
                fold_save = os.path.join(false_folder,select)
            file_move_path = os.path.join(fold_save,os.path.basename(file_save_path))
            os.replace(file_save_path, file_move_path)
        return redirect(url_for('predict_image'))
    return render_template(config.RESPONSE_FILE_HTML, class_name = result['name'], confidence = result['confidence'], class_names = categories)

if __name__ == '__main__':
    app.run(host='0.0.0.0')