#docker run --gpus all -it --rm -v /:/host -v ../Data1/Input:/etri_workspace/Input -v ../Data1/Sample:/etri_workspace/Data -v ../Data1/Output:/etri_workspace/Output etri_scapi:0.1.3
docker run --gpus all -it --rm -v /:/host -v ../VideoData2/Input:/etri_workspace/Input -v ../VideoData2/VideoSample:/etri_workspace/Data -v ../VideoData2/Output:/etri_workspace/Output etri_scapi:0.1.3
