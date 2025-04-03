import ctypes
from misc import *
import numpy as np

dll = ctypes.cdll.LoadLibrary("..\\bin\\SCAPI.dll")
SJCUDARenderer_New = dll.SJCUDARenderer_New
SJCUDARenderer_Initialize = dll.SJCUDARenderer_Initialize
SJCUDARenderer_LoadMPI = dll.SJCUDARenderer_LoadMPI
SJCUDARenderer_InitializeRendering = dll.SJCUDARenderer_InitializeRendering
SJCUDARenderer_Rendering = dll.SJCUDARenderer_Rendering
SJCUDARenderer_Finalize = dll.SJCUDARenderer_Finalize
SJCUDARenderer_CUDA_UCHAR_Alloc = dll.SJCUDARenderer_CUDA_UCHAR_Alloc
SJCUDARenderer_CPU_UCHAR_Alloc = dll.SJCUDARenderer_CPU_UCHAR_Alloc
SJCUDARenderer_CUDA_FLOAT_Alloc = dll.SJCUDARenderer_CUDA_FLOAT_Alloc
SJCUDARenderer_CPU_FLOAT_Alloc = dll.SJCUDARenderer_CPU_FLOAT_Alloc
SJCUDARenderer_CUDA_UCHAR_Free = dll.SJCUDARenderer_CUDA_UCHAR_Free
SJCUDARenderer_CPU_UCHAR_Free = dll.SJCUDARenderer_CPU_UCHAR_Free
SJCUDARenderer_CUDA_FLOAT_Free = dll.SJCUDARenderer_CUDA_FLOAT_Free
SJCUDARenderer_CPU_FLOAT_Free = dll.SJCUDARenderer_CPU_FLOAT_Free

SJCUDARenderer_New.restype = ctypes.c_void_p
SJCUDARenderer_New.argtypes = []

SJCUDARenderer_Initialize.restype = ctypes.c_int
SJCUDARenderer_Initialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]

SJCUDARenderer_LoadMPI.restype = ctypes.c_int
SJCUDARenderer_LoadMPI.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte)]

SJCUDARenderer_InitializeRendering.restype = ctypes.c_int
SJCUDARenderer_InitializeRendering.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]

SJCUDARenderer_Rendering.restype = ctypes.c_int
SJCUDARenderer_Rendering.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte)]

SJCUDARenderer_Finalize.restype = ctypes.c_int
SJCUDARenderer_Finalize.argtypes = [ctypes.c_void_p]

SJCUDARenderer_CUDA_UCHAR_Alloc.restype = ctypes.POINTER(ctypes.c_ubyte)
SJCUDARenderer_CUDA_UCHAR_Alloc.argtypes = [ctypes.c_size_t]

SJCUDARenderer_CPU_UCHAR_Alloc.restype = ctypes.POINTER(ctypes.c_ubyte)
SJCUDARenderer_CPU_UCHAR_Alloc.argtypes = [ctypes.c_size_t]

SJCUDARenderer_CUDA_FLOAT_Alloc.restype = ctypes.POINTER(ctypes.c_float)
SJCUDARenderer_CUDA_FLOAT_Alloc.argtypes = [ctypes.c_size_t]

SJCUDARenderer_CPU_FLOAT_Alloc.restype = ctypes.POINTER(ctypes.c_float)
SJCUDARenderer_CPU_FLOAT_Alloc.argtypes = [ctypes.c_size_t]

SJCUDARenderer_CUDA_UCHAR_Free.restype = None
SJCUDARenderer_CUDA_UCHAR_Free.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]

SJCUDARenderer_CPU_UCHAR_Free.restype = None
SJCUDARenderer_CPU_UCHAR_Free.argtypes = [ctypes.POINTER(ctypes.c_ubyte)]

SJCUDARenderer_CUDA_FLOAT_Free.restype = None
SJCUDARenderer_CUDA_FLOAT_Free.argtypes = [ctypes.POINTER(ctypes.c_float)]

SJCUDARenderer_CPU_FLOAT_Free.restype = None
SJCUDARenderer_CPU_FLOAT_Free.argtypes = [ctypes.POINTER(ctypes.c_float)]


scenedir = "..\\data\\Sample"
poses, pts3d, perm, w2c, c2w, hwf = load_colmap_data(scenedir)
cdepth, idepth = computecloseInfinity(poses, pts3d, perm)
close_depth = np.min(cdepth) * 0.9
inf_depth = np.max(idepth) * 2.0
focal = poses[2, 4, :]

