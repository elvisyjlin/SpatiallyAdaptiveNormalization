# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Demo server"""

from flask import Flask
from flask_cors import CORS
app = Flask(__name__, template_folder='../demo')
CORS(app)

from flask import abort
from flask import jsonify
from flask import render_template
from flask import request

import base64
import json
import numpy as np
import time
import threading
from glob import glob
from os.path import basename, exists, join
from io import BytesIO
from PIL import Image

from .segmentation import decode_image


DATA_ROOT = {}
DEMO_DATASETS = {}
COLORMAP = None
MODEL = None
LOCK = None

@app.before_first_request
def _before_first_request():
    from .model import Model
    from .segmentation import get_colormap_values
    from .preprocess import preprocess_demo_images

    print('Hi, I\'m awake now!')
    
    with open('server/config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    global DATA_ROOT
    DATA_ROOT = config['data_root']
    
    global COLORMAP
    COLORMAP = get_colormap_values()
    
    global MODEL
    MODEL = Model(config['experiment_name'], config['load_epoch'])
    
    global LOCK
    LOCK = threading.Lock()
    
    for dataset, path in DATA_ROOT.items():
        if not exists(join(path, 'demo_images')):
            preprocess_demo_images(dataset, path)
        
        demo_imgs = glob(join(path, 'demo_images/*_img.png'))
        demo_imgs = sorted(demo_imgs, key=lambda x: int(basename(x).split('_')[0]))
        demo_anns = glob(join(path, 'demo_images/*_seg.png'))
        demo_anns = sorted(demo_anns, key=lambda x: int(basename(x).split('_')[0]))
        demo_labs = glob(join(path, 'demo_images/*_lab.npz'))
        demo_labs = sorted(demo_labs, key=lambda x: int(basename(x).split('_')[0]))
        assert len(demo_imgs) == len(demo_anns) == len(demo_labs)
        class_list = json.load(open(join(path, 'demo_images/class_list.json'), 'r', encoding='utf-8'))
        DEMO_DATASETS[dataset] = {}
        DEMO_DATASETS[dataset]['num'] = len(demo_imgs)
        DEMO_DATASETS[dataset]['img'] = demo_imgs
        DEMO_DATASETS[dataset]['seg'] = demo_anns
        DEMO_DATASETS[dataset]['lab'] = demo_labs
        DEMO_DATASETS[dataset]['classes'] = class_list
    
    print('I\'m ready.')

@app.route('/')
def index():
    return 'Hello, GuaGAN!'

@app.route('/demo')
def demo():
    if not exists('../demo/index.html'):
        return render_template('index.html')
    return 'Demo site is not ready.'

@app.route('/main.js')
def main_js():
    return render_template('main.js')

@app.route('/config.js')
def config_js():
    return render_template('config.js')

@app.route('/random/<dataset>')
def random(dataset):
    t0 = time.time()
    dataset = dataset.lower()
    if dataset in DEMO_DATASETS:
        img_idx = np.random.randint(0, DEMO_DATASETS[dataset]['num'])
        with open(DEMO_DATASETS[dataset]['img'][img_idx], 'rb') as f:
            img_dataUrl = u'data:image/png;base64,' + base64.b64encode(f.read()).decode('ascii')
        with open(DEMO_DATASETS[dataset]['seg'][img_idx], 'rb') as f:
            ann_dataUrl = u'data:image/png;base64,' + base64.b64encode(f.read()).decode('ascii')
        lab_array = np.load(DEMO_DATASETS[dataset]['lab'][img_idx])['lab']
        color_list = []
        for label in lab_array:
            color_list.append([
                COLORMAP[label].tolist(),
                DEMO_DATASETS[dataset]['classes'][str(label)]
            ])
        print(lab_array)
        dt = time.time() - t0
        return jsonify({ 'image': img_dataUrl, 'annotation': ann_dataUrl, 'dataset': dataset , 'id': img_idx, 'color_list': color_list, 'response_time': dt })
    return abort(404)

@app.route('/generate', methods=['POST'])
def generate():
    t0 = time.time()
    PNG_DATAURL_PREFIX = 'data:image/png;base64,'
    style_img = request.form['style']
    semantic_img = request.form['semantic']
    if style_img.startswith(PNG_DATAURL_PREFIX):
        style_img = style_img.replace(PNG_DATAURL_PREFIX, '')
        style_img = Image.open(BytesIO(base64.b64decode(style_img)))
    else:
        return abort(400)
    if semantic_img.startswith(PNG_DATAURL_PREFIX):
        semantic_img = semantic_img.replace(PNG_DATAURL_PREFIX, '')
        semantic_img = Image.open(BytesIO(base64.b64decode(semantic_img)))
        semantic_img = decode_image(semantic_img, COLORMAP)
    else:
        return abort(400)
    style_img = np.array(style_img)
    if style_img.shape[2] == 4:
        style_img = style_img[..., :3]
    semantic_img = np.array(semantic_img)
    print(np.unique(semantic_img))
    
    LOCK.acquire()
    gen_t0 = time.time()
    generated_img = MODEL(style_img, semantic_img)
    gen_dt = time.time() - gen_t0
    LOCK.release()
    
    generated_img = Image.fromarray(generated_img)
    buffer = BytesIO()
    generated_img.save(buffer, format='PNG')
    dataUrl = u'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    dt = time.time() - t0
    return jsonify({ 'generated_img': dataUrl, 'response_time': dt, 'generation_time': gen_dt })
