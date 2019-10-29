
#NCCL_DIR = $(HOME)/wsa/2019/nccl/nccl_2.4.8-1+cuda10.1_ppc64le
NCCL_DIR = /usr/workspace/wsb/brain/nccl2/nccl_2.4.8-1+cuda10.1_ppc64le
CXX = g++

all: nccl_test.exe

nccl_test.exe: nccl_test.o
	nvcc -g $< -o $@  -ccbin mpicxx -lnccl -L$(NCCL_DIR)/lib -Xlinker -rpath,$(NCCL_DIR)/lib

nccl_test.o: nccl_test.cpp
	nvcc -I$(NCCL_DIR)/include -ccbin mpicxx -g -std=c++11 -o $@ -c $< -Xcompiler -Wall

run: nccl_test.exe
	jsrun -n1 -r 1 -a 2 -c 10 -g 2 -b packed:5 -d packed ./nccl_test.exe

cuda-memcheck: nccl_test.exe
	jsrun -n1 -r 1 -a 2 -c 10 -g 2 -b packed:5 -d packed cuda-memcheck ./nccl_test.exe

clean:
	rm -f *.o *.exe *.d
