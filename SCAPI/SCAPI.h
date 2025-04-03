#pragma once
#ifdef SC_API_EXPORTS
#define SC_API __declspec(dllexport)
#else
#define SC_API __declspec(dllimport)
#endif

class SJCUDARenderer;
extern "C" {
	SC_API SJCUDARenderer* SJCUDARenderer_New();	
	SC_API int SJCUDARenderer_Initialize(SJCUDARenderer *renderer, size_t* pDim, size_t* pMPIDim, float* pC2WCPU, float* pC2WCUDA, float* pW2CCPU, float* pW2CCUDA, float* pCIFCPU, float* pCIFCUDA);
	SC_API int SJCUDARenderer_LoadMPI(SJCUDARenderer* renderer, unsigned char* pMPICPU, unsigned char* pMPICUDA);
	SC_API int SJCUDARenderer_LoadCompressedMPI(SJCUDARenderer* renderer, unsigned char* pCompressedMPICPU, unsigned char* pCompressedMPICUDA, unsigned char* pCompressedIdxCPU, unsigned char* pCompressedIdxCUDA);
	SC_API int SJCUDARenderer_InitializeRendering(SJCUDARenderer* renderer, size_t outWidth, size_t outHeight, int numBlend);
	SC_API int SJCUDARenderer_Rendering(SJCUDARenderer* renderer, float* pose_arr, unsigned char* pOutImageCUDA, unsigned char* pOutImageCPU);
	SC_API int SJCUDARenderer_Finalize(SJCUDARenderer* renderer);
	SC_API int SJCUDARenderer_LoadMPIFromFolder(char* foldername, size_t* pMPIDim, unsigned char* pMPICUDA);
	SC_API int SJCUDARenderer_LoadDataFromFolder(char* foldername, size_t* pMPIDim, float* pC2W, float* pC2WCUDA, float* pW2C, float* pW2CCUDA, float* pCIF, float* pCIFCUDA);

	SC_API unsigned char* SJCUDARenderer_CUDA_UCHAR_Alloc(size_t size);
	SC_API unsigned char* SJCUDARenderer_CPU_UCHAR_Alloc(size_t size);
	SC_API float* SJCUDARenderer_CUDA_FLOAT_Alloc(size_t size);
	SC_API float* SJCUDARenderer_CPU_FLOAT_Alloc(size_t size);

	SC_API void SJCUDARenderer_CUDA_UCHAR_Free(unsigned char* buffer);
	SC_API void SJCUDARenderer_CPU_UCHAR_Free(unsigned char* buffer);
	SC_API void SJCUDARenderer_CUDA_FLOAT_Free(float* buffer);
	SC_API void SJCUDARenderer_CPU_FLOAT_Free(float* buffer);
}