etriSCAPI: *.cu *.cpp
	nvcc -ccbin g++ -std=c++11 -m64 -I. \
	*.cu *.cpp -o etriSCAPI `pkg-config opencv4 --cflags --libs`

clean: 
	rm etriSCAPI
    
