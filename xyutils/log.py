import os
import logging
import datetime

import colorlog

from .os import mkdir

LOG_COLORS_CONFIG = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


def init_log(log_dir='log', level=logging.INFO, to_file=True, log_file=None):
    if log_file:
        log_dir = os.path.dirname(log_file)
    # 创建一个logging对象
    logger = logging.getLogger()
    logger.handlers = []
    # 创建一个屏幕对象
    sh = logging.StreamHandler()
    # 配置显示格式  可以设置两个配置格式  分别绑定到文件和屏幕上
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        log_colors=LOG_COLORS_CONFIG)
    # formatter = logging.Formatter('[%(levelname)s] %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(level)  # 总开关
    sh.setLevel(level)
    if to_file:
        # 创建一个文件对象  创建一个文件对象,以UTF-8 的形式写入 标配版.log 文件中
        mkdir(log_dir)
        if log_file:
            log_name = log_file
        else:
            f_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log'
            log_name = os.path.join(log_dir, f_name)
        if os.path.exists(log_name):
            os.remove(log_name)
        fh = logging.FileHandler(log_name, encoding='utf-8')
        formatter_f = logging.Formatter(
            '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
        fh.setFormatter(formatter_f)  # 将格式绑定到两个对象上
        logger.addHandler(fh)
        fh.setLevel(level)  # 写入文件的从10开始
        logger.info(f'The log  {log_name} was successfully initialized!')
    return logger
