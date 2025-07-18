import sys
import argparse
import shutil

from loguru import logger
from pathlib import Path

import common
import version as V


def init_data():
    logger.info(f"init_data()")

    # 컨테이너 내부에 항상 고정된 위치에 폴더 위치
    logger.info(f"INPUT_DIR: {common.INPUT_DIR} {Path(common.INPUT_DIR).absolute()}")
    logger.info(f"SCENE_DIR: {common.SCENE_DIR} {Path(common.SCENE_DIR).absolute()}")

    # INPUT_DIR에 입력된 데이터로 폴더 구조 초기화
    path_input = Path(common.INPUT_DIR)
    path_image = Path(common.SCENE_DIR) / "Image" / "images"
    path_image.mkdir(parents=True, exist_ok=True)

    # 현재는 그대로 파일복사 / 필요에 따라서 프레임 구조등으로 변경
    shutil.rmtree(path_image)
    shutil.copytree(path_input, path_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ETRI Spatial Computing Engine v{V.VERSION}")

    try:
        init_data()

        logger.info("Success")
        sys.exit(0)

    except Exception as ex:
        # 에러 발생시 로그 출력하고, exit_code를 0이 아닌 값으로 설정
        logger.error(ex)
        sys.exit(1)
