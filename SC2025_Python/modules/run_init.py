import sys
import argparse
import shutil

from loguru import logger
from pathlib import Path, PurePath

import common
import version as V
import SCDecoder
import json
import cv2
def init_data():
    logger.info(f"init_data()")

    # 컨테이너 내부에 항상 고정된 위치에 폴더 위치
    logger.info(f"INPUT_DIR: {common.INPUT_DIR} {Path(common.INPUT_DIR).absolute()}")
    logger.info(f"SCENE_DIR: {common.SCENE_DIR} {Path(common.SCENE_DIR).absolute()}")
    logger.info(f"SCENE_DIR: {common.OUTPUT_DIR} {Path(common.OUTPUT_DIR).absolute()}")

    # INPUT_DIR에 입력된 데이터로 폴더 구조 초기화
    path_input = Path(common.INPUT_DIR)
    path_output = Path(common.OUTPUT_DIR)

    path_output.mkdir(parents=True, exist_ok=True)

    file_list = list(Path(path_input).iterdir())
    numDecoder = len(file_list)
    
    if file_list[0].suffix == ".png" or file_list[0].suffix == ".PNG" or file_list[0].suffix == ".jpg" or file_list[0].suffix == ".JPG":
        img = cv2.imread(file_list[0])
        data = {}
        for i in range(numDecoder):
            data["cam{:02d}".format(i)] = {}
            data["cam{:02d}".format(i)]["numFrame"] = 1
            data["cam{:02d}".format(i)]["FrameRate"] = 1.0
            data["cam{:02d}".format(i)]["Width"] = img.shape[1]
            data["cam{:02d}".format(i)]["Height"] = img.shape[0]
            
        with open(path_output / "info.json", "w") as file:
            json.dump(data, file, indent=4, sort_keys=True)

        path_image = Path(common.SCENE_DIR) / "Image" / "0" / "images"
        path_image.mkdir(parents=True, exist_ok=True)

        shutil.rmtree(path_image)
        shutil.copytree(path_input, path_image)
    else:
        decoder = SCDecoder.SCDecoder()
        numDecoder = decoder.Initialize(common.INPUT_DIR, common.SCENE_DIR + "/Image")
        data = {}
        for i in range(numDecoder):
            data["cam{:02d}".format(i)] = {}
            data["cam{:02d}".format(i)]["numFrame"] = decoder.GetFrameNumber(i)
            data["cam{:02d}".format(i)]["FrameRate"] = decoder.GetFrameRate(i)
            data["cam{:02d}".format(i)]["Width"] = decoder.GetWidth(i)
            data["cam{:02d}".format(i)]["Height"] = decoder.GetHeight(i)
        with open(path_output / "info.json", "w") as file:
            json.dump(data, file, indent=4, sort_keys=True)
        
        decoder.DoDecoding()
        decoder.Finalize()
    
    # 현재는 그대로 파일복사 / 필요에 따라서 프레임 구조등으로 변경
    

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
