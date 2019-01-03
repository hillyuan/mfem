make config MFEM_CXX="nvcc" \
CXXFLAGS="-g -O0 --restrict --expt-extended-lambda -x=cu -arch=sm_60 -std=c++11 -m64" \
MFEM_EXT_LIBS="-L/usr/local/cuda/lib64-lrt -lcuda -lcudart -lcudadevrt -lnvToolsExt" \
MFEM_DEBUG=YES OPTIM_FLAGS=-O0 MFEM_USE_MM=YES
