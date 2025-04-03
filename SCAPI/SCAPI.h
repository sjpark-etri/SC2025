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
}