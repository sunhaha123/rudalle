import os

# ai工具名称
TOOL_NAME = 'dlm'
# redis配置
REDIS_HOST = 'muse-api.gululu.com'
REDIS_PORT = 6377
REDIS_PSD = 'a9d1as1laA'
REDIS_DB_BROKEN = 10
REDIS_DB_RESULT = 11


class BaseConfig():
    BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
    SECRET_KEY = "abcdefg"


    TEMPLATES_AUTO_RELOAD = True
    SEND_FILE_MAX_AGE_DEFAULT = 1
    
    CELERY_CONFIG = {
        'broker_url': f'redis://:{REDIS_PSD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_BROKEN}',
        'result_backend': f'redis://:{REDIS_PSD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_RESULT}',
        'task_serializer': 'json',
        'result_serializer': 'json',
        'accept_content': ['json'],
        'include': ["task"],
    }
if __name__ == "__main__":
    b = BaseConfig
    print(b.CELERY_CONFIG['broker_url'])