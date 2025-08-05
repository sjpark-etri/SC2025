SET1=$(seq 10 5 50)
for i in $SET1
do
    echo $i
    #docker run --gpus all -it --rm -v ../Data2/Sample:/etri_workspace/Data -v ../Data2/Output:/etri_workspace/Output etri_scapi:0.1.0 python run_render.py --view_range 0.2 --focal $i --num_views 49 --result Result-$i-0.4
    docker run --gpus all -it --rm -v ../Data2/Sample:/etri_workspace/Data -v ../Data2/Output:/etri_workspace/Output etri_scapi:0.1.0 python run_render_quilt.py --view_range 0.8 --focal $i --rows 7 --cols 7 --result $i-0.8-qs7x7.png
done