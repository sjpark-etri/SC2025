#docker run --gpus all -it --rm -v /:/host -v ../Data1/Input:/etri_workspace/Input -v ../Data1/Sample:/etri_workspace/Data -v ../Data1/Output:/etri_workspace/Output etri_scapi:0.1.2
docker run --gpus all -it --rm -v /:/host -v ../VideoData1/Input:/etri_workspace/Input -v ../VideoData1/VideoSample:/etri_workspace/Data -v ../VideoData1/Output:/etri_workspace/Output etri_scapi:0.1.2
