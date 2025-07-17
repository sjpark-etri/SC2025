docker run --gpus all -it --rm -v ../Data/Sample:/etri_workspace/Data/Image -v ../Data/Param:/etri_workspace/Data/Param scapi2 python run_build_param.py
docker run --gpus all -it --rm -v ../Data/Sample:/etri_workspace/Data/Image -v ../Data/Param:/etri_workspace/Data/Param -v ../Data/Layer:/etri_workspace/Data/Layer scapi2 python run_build_layer.py --factor 4
docker run --gpus all -it --rm -v ../Data/Param:/etri_workspace/Data/Param -v ../Data/Layer:/etri_workspace/Data/Layer -v ../Data:/etri_workspace/Data/Quilt scapi2 python run_render_quilt.py --view_range 1.0 --focal 50 --rows 7 --cols 7 --result res.png


