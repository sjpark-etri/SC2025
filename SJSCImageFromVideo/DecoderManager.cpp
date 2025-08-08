#include "DecoderManager.h"
#include "SCDecoder.h"

#include <stdio.h>
#include <string.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;

void OnCallbackForDecoding(unsigned char* frameBuffer, int width, int height, int frameID, int decoderID, void* callerPtr)
{
	((DecoderManager*)callerPtr)->DecodingProcess(frameBuffer, frameID, decoderID);
}

DecoderManager::DecoderManager()
{
	m_pDecoder = NULL;
	m_bStored = NULL;
	m_ppDecoderFrame = NULL;
	m_pWidth = NULL;
	m_pHeight = NULL;
}

DecoderManager::~DecoderManager()
{
	if (m_pDecoder) {
		delete[]m_pDecoder;
		m_pDecoder = NULL;
	}
	if (m_bStored) {
		delete[]m_bStored;
		m_bStored = NULL;
	}
	if (m_ppDecoderFrame) {
		delete[]m_ppDecoderFrame;
		m_ppDecoderFrame = NULL;
	}	
	if(m_pWidth){
		delete []m_pWidth;
		m_pWidth = NULL;
	}
	if(m_pHeight){
		delete []m_pHeight;
		m_pHeight = NULL;
	}
}
void DecoderManager::Initialize(int numDecoder, char** filenames, const char *foldername, int mode = 0)
{
	m_numDecoder = numDecoder;
	strcpy(m_pFoldername, foldername);
	m_mode = mode;
	char filename[2048];
	if (m_mode == 1) {
		for (int i = 0; i < m_numDecoder; i++) {
			sprintf(filename, "mkdir %s/%d", m_pFoldername, i);
			system(filename);
		}
	}
	m_pDecoder = new SCDecoder[m_numDecoder];
	m_bStored = new bool[m_numDecoder];
	m_pWidth = new int[m_numDecoder];
	m_pHeight = new int[m_numDecoder];
	m_ppDecoderFrame = new unsigned char* [m_numDecoder];
	for (int i = 0; i < m_numDecoder; i++) {
		m_pDecoder[i].Initialize(filenames[i], i);
		
		m_pWidth[i] = m_pDecoder[i].GetWidth();
		m_pHeight[i] = m_pDecoder[i].GetHeight();

		m_pDecoder[i].SetSCDecCallBack(OnCallbackForDecoding, this);
		//m_pDecoder[i].StartDecoding(10, 11);
		//m_pDecoder[i].StartDecoding(0, 2);
		m_pDecoder[i].StartDecoding();
		m_bStored[i] = false;
	}
	
}


void DecoderManager::DoDecoding()
{
	char filename[2048];
	int cnt = 0;
	bool allStored;
	Mat img;
	Mat cvtImg;

	while (true) {
		while (true) {
			allStored = true;
			for (int i = 0; i < m_numDecoder; i++) {
				if (!m_pDecoder[i].IsFinish() && !m_bStored[i]) {
					allStored = false;
					break;
				}
			}
			if (allStored) {
				break;
			}

			std::this_thread::yield();
		}
		//
		if (m_mode == 0) {
			sprintf(filename, "mkdir -p %s/%d/images", m_pFoldername, m_iCurrentFrame);
			system(filename);

			for (int i = 0; i < m_numDecoder; i++) {
				img = Mat(m_pHeight[i], m_pWidth[i], CV_8UC3, m_ppDecoderFrame[i]);
				cvtColor(img, cvtImg, COLOR_BGR2RGB);
				sprintf(filename, "%s/%d/images/%03d.png", m_pFoldername, m_iCurrentFrame, i);
				imwrite(filename, cvtImg);
			}
		}
		else if (m_mode == 1) {
			for (int i = 0; i < m_numDecoder; i++) {
				img = Mat(m_pHeight[i], m_pWidth[i], CV_8UC3, m_ppDecoderFrame[i]);
				cvtColor(img, cvtImg, COLOR_BGR2RGB);
				sprintf(filename, "%s/%d/%d.jpg", m_pFoldername, i, m_iCurrentFrame);
				imwrite(filename, cvtImg);
			}
		}
		
		for (int i = 0; i < m_numDecoder; i++) {
			m_bStored[i] = false;
			m_pDecoder[i].ReadyDecoding();
		}

		cnt = 0;
		for (int i = 0; i < m_numDecoder; i++) {
			if (m_pDecoder[i].IsFinish()) {
				cnt++;
			}
		}
		if (cnt == m_numDecoder) break;
	}

	for (int i = 0; i < m_numDecoder; i++) {
		m_pDecoder[i].StopDecoding();
	}
}

void DecoderManager::DecodingProcess(unsigned char* frameBuffer, int frameID, int decoderID)
{
	/*Mat img;
	Mat cvtImg;
	char filename[1024];
	img = Mat(height, width, CV_8UC3, frameBuffer);
	cvtColor(img, cvtImg, COLOR_BGR2RGB);
	sprintf_s(filename, "%s\\%d.png", m_pFoldername, frameID);
	imwrite(filename, cvtImg);*/
	m_bStored[decoderID] = true;
	m_ppDecoderFrame[decoderID] = frameBuffer;
	m_iCurrentFrame = frameID;
}

int64_t DecoderManager::GetNumFrame(int idx)
{
	return m_pDecoder[idx].GetNumFrame();
}

float DecoderManager::GetFrameRate(int idx)
{
	return m_pDecoder[idx].GetFrameRate();
}

int DecoderManager::GetWidth(int idx)
{
	return m_pWidth[idx];
}
int DecoderManager::GetHeight(int idx)
{
	return m_pHeight[idx];
}