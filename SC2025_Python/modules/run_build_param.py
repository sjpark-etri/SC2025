import sys
import argparse
from loguru import logger
from pathlib import Path
import os
import subprocess

import common
import version as V


def build_param():
    # 컨테이너 내부에 항상 고정된 위치에 폴더 위치
    # pathlib.Path를 사용하시면, 윈도우/리눅스에서 경로 처리가 간단합니다.
    
    #logger.info(f"PARAM_DIR: {common.PARAM_DIR} {Path(common.PARAM_DIR).absolute()}")
    #logger.info(f"IMAGE_DIR: {common.IMAGE_DIR} {Path(common.IMAGE_DIR).absolute()}")
    #logger.info(f"LAYER_DIR: {common.LAYER_DIR} {Path(common.LAYER_DIR).absolute()}")
    
    if not os.path.exists(os.path.join(common.PARAM_DIR, 'sparse')):
        os.makedirs(os.path.join(common.PARAM_DIR, 'sparse'))
    
    subprocess.run(['colmap', 'feature_extractor', '--database_path', os.path.join(common.PARAM_DIR, 'database.db'), '--image_path', os.path.join(common.IMAGE_DIR, 'images')], check=True)
    subprocess.run(['colmap', 'exhaustive_matcher', '--database_path', os.path.join(common.PARAM_DIR, 'database.db')], check=True)
    subprocess.run(['colmap', 'mapper','--database_path', os.path.join(common.PARAM_DIR, 'database.db'), '--image_path', os.path.join(common.IMAGE_DIR, 'images'), '--output_path', os.path.join(common.PARAM_DIR, 'sparse')], check=True)            
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ETRI Spatial Computing Engine v{V.VERSION}")   
    
    try:
        build_param()
    
        logger.info("Success")
        sys.exit(0)

    except Exception as ex:
        # 에러 발생시 로그 출력하고, exit_code를 0이 아닌 값으로 설정
        logger.error(ex)
        sys.exit(1)
