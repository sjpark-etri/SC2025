import sys
import argparse
from loguru import logger
from pathlib import Path

import common
import version as V
import SCAPI
import cv2

def render(view_range: float, focal: float, num_views: int, result : str):
    logger.info(f"action_build_param()")
    logger.info(f"view_range : {view_range}")
    logger.info(f"focal      : {focal}")
    logger.info(f"number of views       : {num_views}")
    logger.info(f"result foldername : {result}")
    # 컨테이너 내부에 항상 고정된 위치에 폴더 위치
    # pathlib.Path를 사용하시면, 윈도우/리눅스에서 경로 처리가 간단합니다.
    
    # SCAPI 호출등 엔진 기능 구현
    scene_dir = Path(common.SCENE_DIR)

    api = SCAPI.SCAPI()
    m = api.SetInputFolder(scene_dir / 'Param', scene_dir / 'Layer')
    imgs = api.FullRendering(view_range, focal, num_views)
    res_folder = Path(common.OUTPUT_DIR) / result 
    res_folder.mkdir(parents=True, exist_ok=True)

    for i in range(num_views):
        cv2.imwrite(res_folder  / "{:03d}.png".format(i), imgs[i,...])
    
    api.Finalize()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ETRI Spatial Computing Engine v{V.VERSION}")
    parser.add_argument('--view_range', type=float, required=True, help='Specify view_range.')
    parser.add_argument('--focal', type=float, required=True, help='Specify focal.')
    parser.add_argument('--num_views', type=int, required=True, help='Specify number of views.')
    parser.add_argument('--result', type=str, required=True, help='Specify result foldername.')
    args = parser.parse_args()

    try:
        render(**vars(args))

        logger.info("Success")
        sys.exit(0)

    except Exception as ex:
        # 에러 발생시 로그 출력하고, exit_code를 0이 아닌 값으로 설정
        logger.error(ex)
        sys.exit(1)
