/cluster/home/sejinkim/miniconda3/envs/atlasnet/bin/nvcc -I/cluster/home/sejinkim/miniconda3/envs/atlasnet/lib/python3.6/site-packages/torch/include -I/cluster/home/sejinkim/miniconda3/envs/atlasnet/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/cluster/home/sejinkim/miniconda3/envs/atlasnet/lib/python3.6/site-packages/torch/include/TH -I/cluster/home/sejinkim/miniconda3/envs/atlasnet/lib/python3.6/site-packages/torch/include/THC -I/cluster/home/sejinkim/miniconda3/envs/atlasnet/include/python3.6m -c -c /cluster/home/sejinkim/projects/AtlasNet/auxiliary/ChamferDistancePytorch/chamfer3D/chamfer3D.cu -o /cluster/home/sejinkim/projects/AtlasNet/build/temp.linux-x86_64-3.6/auxiliary/ChamferDistancePytorch/chamfer3D/chamfer3D.o -DCUDA_NO_HALF_OPERATORS -DCUDA_NO_HALF_CONVERSIONS -DCUDA_NO_BFLOAT16_CONVERSIONS -DCUDA_NO_HALF2_OPERATORS --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=chamfer_3D -D_GLIBCXX_USE_CXX11_ABI=0 -gencode=arch=compute_61,code=compute_61 -gencode=arch=compute_61,code=sm_61 -std=c++14