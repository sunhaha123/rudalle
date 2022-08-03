from flask import Blueprint, jsonify, request, send_from_directory
from googletrans import Translator
from uuid import uuid4
from task.rdl import make_image_task
from service.rdl import DdService
from flask_cors import cross_origin

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pre import *
from conf import BaseConfig
import random
import redis


REDIS_HOST = 'muse-api.gululu.com'
REDIS_PORT = 6377
REDIS_PSD = 'a9d1as1laA'
REDIS_DB_BROKEN = 10

conn = redis.Redis(host=REDIS_HOST, port= REDIS_PORT, db=REDIS_DB_BROKEN, password=REDIS_PSD, health_check_interval=30,socket_keepalive=True)

bp_api = Blueprint('bp_api', __name__, url_prefix='/api')

def rank_task(conn,taskid):
    key_search = None
    if conn.hlen('unacked')>0:
        for each in conn.hgetall('unacked'):
            dict_str = conn.hget('unacked',each).decode()
            dict1 = json.loads(dict_str)
            if taskid in dict1[0]['headers']['argsrepr']:
                key_search =  each
                # print('true')

    if key_search is not None:
            index = conn.zrank('unacked_index', key_search)
            return index + 1
    else:
                # print('false')
            return conn.llen('celery')+conn.hlen('unacked')



@bp_api.get("/make_image_v1")
@cross_origin(supports_credentials=True)
def api_make_image_v1():
    '''
    生成图片接口
    '''
    text = request.args.get('text')
    style = request.args.get('style')
    dict_style = {0:'',1:'Chinese ink painting', 2:'Aka mike winkelman, high-definition, Unreal Engine', 3:'Jacek Yerka, Rene Magritte, Igor Morski', 4:'Simon Stalenhag, Ross Tran, Liam WongJohn Harris, Thomas Kinkade', 5:'Pascal Campion'}
    style_en  = dict_style[int(style)]

    translator = Translator(service_urls=['translate.google.cn'])
    if style == 3 or 4:
        word_count = len(style_en.split(','))
        random_num = random.randint(0,word_count-1)
        style_en = style_en.split(',')[random_num]

    text_result = translator.translate(text, dest='en', src='auto').text + ', ' + style_en

    task_id = uuid4().hex
    batchNum = task_id

    make_image_task.delay(text_result,batchNum)
    data = {'taskId':task_id,'prompt_tanslation':text_result,'queue_num':conn.llen('dd')+conn.hlen('unacked')}
    return jsonify(data)


@bp_api.get("/query_progress")
@cross_origin(supports_credentials=True)
def api_query_progress():
    '''
    查询进度接口
    '''
    task_id = request.args.get('task_id')
    if not task_id:
        return jsonify(msg="invalid task_id"), 400
    progress =  DdService.get_progress(task_id)
    if progress == 0:
        img_url = None
    else:
        filename = DdService.get_progress_img_filename(task_id, progress)
        img_url = f"/api/img/task/{filename}"

    queue_num = rank_task(conn,task_id)
    if progress>0:
        queue_num = 0
    return jsonify(progress=progress, progress_img=img_url, queue_num=queue_num)


@bp_api.get("/img/task/<filename>")
@cross_origin(supports_credentials=True)
def get_task_file(filename):
    task_id = filename.split('_')[0]
    task_dir = os.path.join(BaseConfig.BASE_DIR,'images_out','TimeToDisco','partials',task_id)
    img_path=os.path.join(task_dir, filename)
    print(img_path)
    if os.path.exists(img_path):
        return send_from_directory(task_dir, filename, as_attachment=True)
    else:
        return 'file not found', 404

@bp_api.get("/settings/task/<task_id>")
@cross_origin(supports_credentials=True)
def get_task_setting(task_id):
    task_dir = os.path.join(BaseConfig.BASE_DIR,'images_out','TimeToDisco','partials')
    setting_path=os.path.join(task_dir, '{}_settings.txt'.format(task_id))
    print(setting_path)
    if os.path.exists(setting_path):
        return send_from_directory(task_dir, '{}_settings.txt'.format(task_id), as_attachment=True)
    else:
        return 'file not found', 404