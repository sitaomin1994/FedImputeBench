import os

from loguru import logger
import sys


def setup_logger(logging_path, to_file=False, level="INFO"):
    # 移除之前的所有处理器
    logger.remove()
    if to_file:
        # 添加文件输出
        logger.add(
            os.path.join(logging_path, ".log"), format="{time:HH:mm:ss} |{level} | {function}:{line} - {message}",
            level=level
        )
    else:
        # 添加控制台输出
        logger.add(sys.stderr, level=level)
