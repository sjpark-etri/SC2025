import sys
import argparse
from loguru import logger
from pathlib import Path
import subprocess
import common
import version as V

def build_layer(factor: int):
    logger.info(f"action_build_layer()")

    subprocess.run(['make_layer', common.PARAM_DIR, common.IMAGE_DIR, common.LAYER_DIR, str(factor)], check=True)
    #raise Exception("Data file is not exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"ETRI Spatial Computing Engine v{V.VERSION}")
    parser.add_argument('--factor', type=str, required=True, help='Specify factor.')

    args = parser.parse_args()
    factor = args.factor

    try:
        build_layer(factor)

        logger.info("Success")
        sys.exit(0)

    except Exception as ex:
        # 에러 발생시 로그 출력하고, exit_code를 0이 아닌 값으로 설정
        logger.error(ex)
        sys.exit(1)
