import sys
import argparse
from loguru import logger
from pathlib import Path

import common
import version as V
import SCAPI
import cv2
import os

def render_quilt(view_range: float, focal: float, rows: int, cols: int, result : str):
    logger.info(f"action_build_param()")
    logger.info(f"view_range : {view_range}")
    logger.info(f"focal      : {focal}")
    logger.info(f"rows       : {rows}")
    logger.info(f"cols       : {cols}")
    logger.info(f"result filename : {result}")
    # 컨테이너 내부에 항상 고정된 위치에 폴더 위치
    # pathlib.Path를 사용하시면, 윈도우/리눅스에서 경로 처리가 간단합니다.
    
    # SCAPI 호출등 엔진 기능 구현
    api = SCAPI.SCAPI()
    m = api.SetInputFolder(common.PARAM_DIR, common.LAYER_DIR)
    q = api.MakeQuiltImage(view_range, focal, rows, cols)
    cv2.imwrite(os.path.join(common.QUILT_DIR, result), q)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ETRI Spatial Computing Engine v{V.VERSION}")
    parser.add_argument('--view_range', type=float, required=True, help='Specify view_range.')
    parser.add_argument('--focal', type=float, required=True, help='Specify focal.')
    parser.add_argument('--rows', type=int, required=True, help='Specify rows.')
    parser.add_argument('--cols', type=int, required=True, help='Specify cols.')
    parser.add_argument('--result', type=str, required=True, help='Specify result filename.')
    args = parser.parse_args()

    try:
        render_quilt(**vars(args))

        logger.info("Success")
        sys.exit(0)

    except Exception as ex:
        # 에러 발생시 로그 출력하고, exit_code를 0이 아닌 값으로 설정
        logger.error(ex)
        sys.exit(1)
