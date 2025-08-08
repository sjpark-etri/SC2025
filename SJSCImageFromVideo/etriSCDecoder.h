#pragma once
#include <cstdint>
#include <cstddef>
class DecoderManager;
extern "C" {
	DecoderManager* DecoderManager_New();
	int DecoderManager_Initialize(DecoderManager* manager, int numDecoder, char** filenames, const char *foldername);
	int DecoderManager_DoDecoding(DecoderManager* manager);
	int DecoderManager_Finalize(DecoderManager* manager);
	int64_t DecoderManager_GetNumFrame(DecoderManager* manager, int idx);
	float DecoderManager_GetFrameRate(DecoderManager* manager, int idx);
	int DecoderManager_GetWidth(DecoderManager* manager, int idx);
	int DecoderManager_GetHeight(DecoderManager* manager, int idx);

}
