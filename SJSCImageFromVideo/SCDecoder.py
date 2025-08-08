import os
import ctypes
import sys
import numpy as np
import subprocess

class SCDecoder:
    def __init__(self):
        if sys.platform == "win32":
            dll = ctypes.cdll.LoadLibrary("..\\bin\\scdecoder.dll")
        else:
            #os.environ['LD_LIBRARY_PATH'] = "./" + os.environ.get("LD_LIBRARY_PATH", "")
            #dll = ctypes.cdll.LoadLibrary("./libscdecoder.so")
            os.environ['LD_LIBRARY_PATH'] = "/etri_workspace" + os.environ.get("LD_LIBRARY_PATH", "")
            dll = ctypes.cdll.LoadLibrary("/etri_workspace/libscdecoder.so")
        
        self.DecoderManager_New = dll.DecoderManager_New
        self.DecoderManager_New.restype = ctypes.c_void_p
        self.DecoderManager_New.argtypes = []

        self.DecoderManager_Initialize = dll.DecoderManager_Initialize
        self.DecoderManager_Initialize.restype = ctypes.c_int
        self.DecoderManager_Initialize.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        
        self.DecoderManager_DoDecoding = dll.DecoderManager_DoDecoding
        self.DecoderManager_DoDecoding.restype = ctypes.c_int
        self.DecoderManager_DoDecoding.argtypes = [ctypes.c_void_p]

        self.DecoderManager_Finalize = dll.DecoderManager_Finalize
        self.DecoderManager_Finalize.restype = ctypes.c_int
        self.DecoderManager_Finalize.argtypes = [ctypes.c_void_p]

        self.DecoderManager_GetNumFrame = dll.DecoderManager_GetNumFrame
        self.DecoderManager_GetNumFrame.restype = ctypes.c_int64
        self.DecoderManager_GetNumFrame.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self.DecoderManager_GetFrameRate = dll.DecoderManager_GetFrameRate
        self.DecoderManager_GetFrameRate.restype = ctypes.c_float
        self.DecoderManager_GetFrameRate.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def Initialize(self, input_dir, output_dir, width, height, mode):
        self.manager = self.DecoderManager_New()
        dir_list = os.listdir(input_dir)        
        numDecoder = len(dir_list)

        foldername = output_dir.encode('utf-8')
        filenames = [os.path.join(input_dir, d).encode('utf-8') for d in dir_list]
        filenames_arr = (ctypes.c_char_p * len(filenames))(*filenames)
        self.DecoderManager_Initialize(self.manager, numDecoder, filenames_arr, foldername, width, height, mode)
        return numDecoder
    
    def GetFrameNumber(self, idx):
        return self.DecoderManager_GetNumFrame(self.manager, idx)

    def GetFrameRate(self, idx):
        return self.DecoderManager_GetFrameRate(self.manager, idx)
    
    def DoDecoding(self):
        self.DecoderManager_DoDecoding(self.manager)
    
    def Finalize(self):
        self.DecoderManager_Finalize(self.manager)
