import os
import ctypes
import sys
from misc import *
import numpy as np

class SCAPI:
    def __init__(self):
        if sys.platform == "win32":
            dll = ctypes.cdll.LoadLibrary("..\\bin\\SCAPI.dll")
        else:
            os.environ['LD_LIBRARY_PATH'] = "../etriSCAPI" + os.environ.get("LD_LIBRARY_PATH", "")
            dll = ctypes.cdll.LoadLibrary("../etriSCAPI/libetriSCAPI.so")

        self.SJCUDARenderer_New = dll.SJCUDARenderer_New
        self.SJCUDARenderer_New.restype = ctypes.c_void_p
        self.SJCUDARenderer_New.argtypes = []

        self.SJCUDARenderer_Initialize = dll.SJCUDARenderer_Initialize
        self.SJCUDARenderer_Initialize.restype = ctypes.c_int
        self.SJCUDARenderer_Initialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

        self.SJCUDARenderer_LoadMPI = dll.SJCUDARenderer_LoadMPI
        self.SJCUDARenderer_LoadMPI.restype = ctypes.c_int
        self.SJCUDARenderer_LoadMPI.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte)]

        self.SJCUDARenderer_InitializeRendering = dll.SJCUDARenderer_InitializeRendering
        self.SJCUDARenderer_InitializeRendering.restype = ctypes.c_int
        self.SJCUDARenderer_InitializeRendering.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]

        self.SJCUDARenderer_Rendering = dll.SJCUDARenderer_Rendering
        self.SJCUDARenderer_Rendering.restype = ctypes.c_int
        self.SJCUDARenderer_Rendering.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte)]

        self.SJCUDARenderer_Finalize = dll.SJCUDARenderer_Finalize
        self.SJCUDARenderer_Finalize.restype = ctypes.c_int
        self.SJCUDARenderer_Finalize.argtypes = [ctypes.c_void_p]

        self.SJCUDARenderer_CUDA_UCHAR_Alloc = dll.SJCUDARenderer_CUDA_UCHAR_Alloc
        self.SJCUDARenderer_CUDA_UCHAR_Alloc.restype = ctypes.POINTER(ctypes.c_ubyte)
        self.SJCUDARenderer_CUDA_UCHAR_Alloc.argtypes = [ctypes.c_size_t]

        self.SJCUDARenderer_CPU_UCHAR_Alloc = dll.SJCUDARenderer_CPU_UCHAR_Alloc
        self.SJCUDARenderer_CPU_UCHAR_Alloc.restype = ctypes.POINTER(ctypes.c_ubyte)
        self.SJCUDARenderer_CPU_UCHAR_Alloc.argtypes = [ctypes.c_size_t]

        self.SJCUDARenderer_CUDA_FLOAT_Alloc = dll.SJCUDARenderer_CUDA_FLOAT_Alloc
        self.SJCUDARenderer_CUDA_FLOAT_Alloc.restype = ctypes.POINTER(ctypes.c_float)
        self.SJCUDARenderer_CUDA_FLOAT_Alloc.argtypes = [ctypes.c_size_t]

        self.SJCUDARenderer_CPU_FLOAT_Alloc = dll.SJCUDARenderer_CPU_FLOAT_Alloc
        self.SJCUDARenderer_CPU_FLOAT_Alloc.restype = ctypes.POINTER(ctypes.c_float)
        self.SJCUDARenderer_CPU_FLOAT_Alloc.argtypes = [ctypes.c_size_t]

        self.SJCUDARenderer_CUDA_UCHAR_Free = dll.SJCUDARenderer_CUDA_UCHAR_Free
        self.SJCUDARenderer_CUDA_UCHAR_Free.restype = None
        self.SJCUDARenderer_CUDA_UCHAR_Free.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]

        self.SJCUDARenderer_CPU_UCHAR_Free = dll.SJCUDARenderer_CPU_UCHAR_Free
        self.SJCUDARenderer_CPU_UCHAR_Free.restype = None
        self.SJCUDARenderer_CPU_UCHAR_Free.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]

        self.SJCUDARenderer_CUDA_FLOAT_Free = dll.SJCUDARenderer_CUDA_FLOAT_Free
        self.SJCUDARenderer_CUDA_FLOAT_Free.restype = None
        self.SJCUDARenderer_CUDA_FLOAT_Free.argtypes = [ctypes.POINTER(ctypes.c_float)]

        self.SJCUDARenderer_CPU_FLOAT_Free = dll.SJCUDARenderer_CPU_FLOAT_Free
        self.SJCUDARenderer_CPU_FLOAT_Free.restype = None
        self.SJCUDARenderer_CPU_FLOAT_Free.argtypes = [ctypes.POINTER(ctypes.c_float)]

        self.SJCUDARenderer_LoadMPIFromFolder = dll.SJCUDARenderer_LoadMPIFromFolder
        self.SJCUDARenderer_LoadMPIFromFolder.restype = ctypes.c_int
        self.SJCUDARenderer_LoadMPIFromFolder.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_ubyte)]

        self.SJCUDARenderer_LoadDataFromFolder = dll.SJCUDARenderer_LoadDataFromFolder
        self.SJCUDARenderer_LoadDataFromFolder.restype = ctypes.c_int
        self.SJCUDARenderer_LoadDataFromFolder.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

        self.SJCUDARenderer_GetRenderPath = dll.SJCUDARenderer_GetRenderPath
        self.SJCUDARenderer_GetRenderPath.restype = ctypes.POINTER(ctypes.c_float)
        self.SJCUDARenderer_GetRenderPath.argtypes = [ctypes.c_char_p]
    
    def GetRenderPath(self, path, path_filename):
        numView = 45
        filename = os.path.join(path, path_filename).encode('utf-8')
        return numView, self.SJCUDARenderer_GetRenderPath(filename)
    def GetRenderPath(self, path):
        poses, pts3d, perm, w2c, c2w, hwf = load_colmap_data(path)
        cdepth, idepth = computecloseInfinity(poses, pts3d, perm)
        close_depth = np.min(cdepth) * 0.9
        inf_depth = np.max(idepth) * 2.0
        render_poses = generate_render_path_param1(poses, close_depth, inf_depth, 1.0, 10.0, comps=[True, False, False], N=49)

        render_poses = np.concatenate([render_poses[...,1:2], -render_poses[...,0:1], render_poses[...,2:]], -1)

        render_poses = np.transpose(render_poses, (0, 2, 1))
        render_poses = render_poses[:,0:4,:]
        bottom_column = np.tile(np.array([0, 0, 0, 1]).reshape(1, 4, 1), (49, 1, 1))
        render_poses = np.concatenate([render_poses, bottom_column], axis=2)

        numView = render_poses.shape[0]
        render_poses = render_poses.reshape(-1).astype(np.float32)
        pViewArr = render_poses.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return numView, pViewArr
    

