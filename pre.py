from flask import Flask, Response, request, render_template,jsonify
from flask_cors import cross_origin

ALLOWED_MEMORY = 6.0  # choose your GPU memory in GB, min value 3.5GB
if ALLOWED_MEMORY < 4.5:
    DALLE_BS = 1
elif ALLOWED_MEMORY < 5.5:
    DALLE_BS = 2
elif ALLOWED_MEMORY < 6.5:
    DALLE_BS = 3
elif ALLOWED_MEMORY < 7.5:
    DALLE_BS = 4
elif ALLOWED_MEMORY < 8.5:
    DALLE_BS = 5
elif ALLOWED_MEMORY < 9.5:
    DALLE_BS = 6
elif ALLOWED_MEMORY < 10.5:
    DALLE_BS = 7
else:
    DALLE_BS = 8

if ALLOWED_MEMORY < 6.0:
    USE_SUPER_RES = False
else:
    USE_SUPER_RES = True

print('ruDALL-E batch size:', DALLE_BS)
print('super-resolution:', USE_SUPER_RES)

# import multiprocessing and torch
import multiprocessing
import torch
from psutil import virtual_memory

ram_gb = round(virtual_memory().total / 1024 ** 3, 1)

print('CPU:', multiprocessing.cpu_count())
print('RAM GB:', ram_gb)
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device.type)

# import transformers, tools, and translation
import transformers
import more_itertools
import tqdm
from googletrans import Translator
import ruclip
from rudalle.pipelines import generate_images, show, super_resolution
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything, torch_tensors_to_pil_list

# prepare models:
# dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
device = 'cuda'
tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)  # for stable generations you should use dwt=False

# prepare utils:
clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=8)


# generation code
def generate_codebooks(text, tokenizer, dalle, top_k, top_p, images_num, image_prompts=None, temperature=1.0, bs=8,
                       seed=None, use_cache=True):
    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')
    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    codebooks = []
    for chunk in more_itertools.chunked(range(images_num), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
            out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
            has_cache = False
            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.repeat(chunk_bs, 1)
            for idx in tqdm(range(out.shape[1], total_seq_length)):
                idx -= text_seq_length
                if image_prompts is not None and idx in prompts_idx:
                    out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                else:
                    logits, has_cache = dalle(out, attention_mask,
                                              has_cache=has_cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, vocab_size:]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    out = torch.cat((out, sample), dim=-1)
            codebooks.append(out[:, -image_seq_length:].cpu())
    return codebooks

from qiniu import Auth, put_file
import os
QINIU_AK = "bRoN9YvCIIQfLE47QiPBEWydi94jaqK5eoHmQs1X"
QINIU_SK = "Ig2uIrCsFAO_uSBP4OQZzTfHS-nL27CEK3WLOOfQ"
QINIU_BUCKET = "rex-qn"
IMAGE_EXT = "png"
def uplaod_task_img(task_id: str, progress: int, img_local_path: str):
    '''
    上传任务进度图片
    若无法获取进度, 请在生成完成后上传最终图片, 进度为100

    参数:
    task_id: 任务id
    progress: 生成进度, 1-100,
    img_local_path: 本地图片路径
    '''

    if not os.path.exists(img_local_path):
        print(f'file_does_not_exist: {img_local_path}')
        return False
    auth = Auth(QINIU_AK, QINIU_SK)
    save_name = f"muse/{task_id}/{progress}.{IMAGE_EXT}"
    token = auth.upload_token(QINIU_BUCKET, save_name)
    ret, info = put_file(token, save_name, img_local_path, version='v2')
    print("upload_task_img_done: ", info)
    if ret and ret.get('key') == save_name:
        return True
    return False

def _get_img_url(task_id: str, progress: int):
    '''
    拼接图片url
    '''
    return os.path.join(BaseConfig.QINIU_URL, f"muse/{task_id}/{progress}.png")

class BaseConfig():
    BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    SECRET_KEY = "3k9n21d903"

    TEMPLATES_AUTO_RELOAD = True
    SEND_FILE_MAX_AGE_DEFAULT = 1

    # SQLALCHEMY_ECHO = True
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_DATABASE_URI = 'mysql://muse_test:MuseTest123@rm-uf6lbd0zmabc6gy6tko.mysql.rds.aliyuncs.com:3306/muse_test'

    QINIU_URL = 'http://rex-qn.gululu-a.com/'

    REDIS_PROGRESS_DB = {
        'host': 'muse-api.gululu.com',
        'port': 6377,
        'psd': 'a9d1as1laA',
        'db': 12,
    }

    CELERY_CONFIG = {
        'broker_url': 'redis://:a9d1as1laA@muse-api.gululu.com:6377/10',
        'result_backend': 'redis://:a9d1as1laA@muse-api.gululu.com:6377/11',
        'task_serializer': 'json',
        'result_serializer': 'json',
        'accept_content': ['json'],
        'include': ["main.task"]
    }

from uuid import uuid4
import redis

REDIS_HOST = 'muse-api.gululu.com'
REDIS_PORT = 6377
REDIS_PSD = 'a9d1as1laA'
REDIS_DB_BROKEN = 10
conn = redis.Redis(host=REDIS_HOST, port= REDIS_PORT, db=REDIS_DB_BROKEN, password=REDIS_PSD, health_check_interval=30,socket_keepalive=True)

app = Flask(__name__)

@app.route("/make_image_v1", methods=["GET"])
@cross_origin(supports_credentials=True)
def api_make_image_v1():
    '''
    生成图片接口
    '''
    text = request.args.get('text')
    num_items = 4  # @param {type:"slider", min:2, max:12, step:2}
    model_name = 'Malevich'  # @param ['Malevich', 'Surrealist', 'Malevich_v2', 'Kandinsky', 'Emojich']
    total_num = num_items * 3

    # translation
    translator = Translator(service_urls=['translate.google.cn'])
    text = translators.google(text, from_language='en', to_language='ru')

    # build rudall model
    dalle = get_rudalle_model(model_name, pretrained=True, fp16=True, device='cuda')

    import random
    seed_everything(random.randint(1, 2 ** 32 - 1))
    pil_images = []
    scores = []

    for top_k, top_p, images_num in [
        (2048, 0.995, 9),
        # (1536, 0.99, 1),
        # (1024, 0.99, 8),
        # (1024, 0.98, 8),
        # (512, 0.97, 8),
        # (384, 0.96, 8),
        # (256, 0.95, 8),
        # (128, 0.95, 8),
    ]:
        _pil_images, _scores = generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, bs=4,
                                               top_p=top_p)
        pil_images += _pil_images
        scores += _scores
    idx = uuid4().hex
    for i in range(len(pil_images)):
        task_id = idx + str(i)
        pil_images[i].save(f"./result/{task_id}.png")
        uplaod_task_img(task_id, 100, f"./result/{task_id}.png")

    img_urls = []
    for i in range(len(pil_images)):
        task_id = idx + str(i)
        img_urls.append(_get_img_url(task_id, 100))

    data = {'taskId':task_id,'prompt_tanslation':text,'imgs':img_urls}
    return jsonify(data)

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=6006, debug=True)