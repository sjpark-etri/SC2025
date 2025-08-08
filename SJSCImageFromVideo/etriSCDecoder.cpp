#include "etriSCDecoder.h"
#include "DecoderManager.h"
DecoderManager* DecoderManager_New()
{
	return new DecoderManager();
}
int DecoderManager_Initialize(DecoderManager* manager, int numDecoder, char** filenames, const char *foldername)
{
	manager->Initialize(numDecoder, filenames, foldername, 0);
	return 0;
}	
int DecoderManager_DoDecoding(DecoderManager* manager)
{
	manager->DoDecoding();
	return 0;
}
int DecoderManager_Finalize(DecoderManager* manager)
{
	delete manager;
	return 0;
}
int64_t DecoderManager_GetNumFrame(DecoderManager* manager, int idx)
{
	return manager->GetNumFrame(idx);
}
float DecoderManager_GetFrameRate(DecoderManager* manager, int idx)
{
	return manager->GetFrameRate(idx);
}
int DecoderManager_GetWidth(DecoderManager* manager, int idx)
{
	return manager->GetWidth(idx);
}
int DecoderManager_GetHeight(DecoderManager* manager, int idx)
{
	return manager->GetHeight(idx);
}
