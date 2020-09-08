// clang-format off
/*
// This code uses control code derived from code with the following license:
//
// Copyright (c) 2019-2020 Ben Ashbaugh
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
*/
// clang-format on

#include <CL/cl2.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <cmath>

#define CHECK_ERROR(err) do {			\
    if (err != CL_SUCCESS) {			\
      printf("error %d\n", err);		\
      assert(false);				\
    }						\
  } while(false)				\

static std::vector<cl_uchar> read_file(const std::string &filename) {
  std::ifstream is(filename, std::ios::binary);
  std::vector<cl_uchar> ret;
  if (!is.good()) {
    return ret;
  }

  size_t filesize = 0;
  is.seekg(0, std::ios::end);
  filesize = (size_t)is.tellg();
  is.seekg(0, std::ios::beg);

  ret.reserve(filesize);
  ret.insert(ret.begin(), std::istreambuf_iterator<char>(is),
             std::istreambuf_iterator<char>());

  return ret;
}

void parse_arguments(int argc, char **argv, int &platform_index,
                     int &device_index) {
  bool printUsage = false;

  if (argc < 1) {
    printUsage = true;
  } else {
    for (size_t i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-d")) {
        if (++i < argc) {
          device_index = strtol(argv[i], NULL, 10);
        }
      } else if (!strcmp(argv[i], "-p")) {
        if (++i < argc) {
          platform_index = strtol(argv[i], NULL, 10);
        }
      } else {
        printUsage = true;
      }
    }
  }
  if (printUsage) {
    fprintf(stderr,
            "Usage: saxpy [options]\n"
            "Options:\n"
            "      -d: Device Index (default = %d)\n"
            "      -p: Platform Index (default = %d)\n",
            device_index, platform_index);

    exit(-1);
  }
}

int main(int argc, char **argv) {
  int platform_index = 0;
  int device_index = 0;

  parse_arguments(argc, argv, platform_index, device_index);

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  printf("Running on platform: %s\n",
         platforms[platform_index].getInfo<CL_PLATFORM_NAME>().c_str());

  std::vector<cl::Device> devices;
  platforms[platform_index].getDevices(CL_DEVICE_TYPE_ALL, &devices);

  printf("Running on device: %s\n",
         devices[device_index].getInfo<CL_DEVICE_NAME>().c_str());
  printf("CL_DEVICE_ADDRESS_BITS is %d for this device.\n",
         devices[device_index].getInfo<CL_DEVICE_ADDRESS_BITS>());

  cl_int err = CL_SUCCESS;
  cl::Context context(devices[device_index]);
  cl::CommandQueue queue(context, devices[device_index], err);
  CHECK_ERROR(err);

#if !defined(USE_SPIRV) || USE_SPIRV == 1
  const char *filename =
      (sizeof(void *) == 8) ? "saxpy_kernel64.spv" : "saxpy_kernel32.spv";
  std::vector<cl_uchar> spirv = read_file(filename);
  cl::Program program(clCreateProgramWithIL(context(), spirv.data(), spirv.size(), &err));
#else
  const char *filename = "saxpy_kernel.cl";
  std::vector<cl_uchar> src = read_file(filename);
  std::string srcs((const char *)src.data(), src.size());
  cl::Program program(context, srcs, false, &err);
#endif
  CHECK_ERROR(err);

  const char *build_options = NULL;
  program.build(build_options);

  for (auto &device : program.getInfo<CL_PROGRAM_DEVICES>()) {
    printf("Program build log for device %s:\n",
           device.getInfo<CL_DEVICE_NAME>().c_str());
    printf("%s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
  }

  const char *kernel_name = "saxpy";
  cl::Kernel kernel = cl::Kernel(program, kernel_name, &err);
  CHECK_ERROR(err);

  size_t num_elements = 1 << 20;
  size_t buffer_size = num_elements * sizeof(cl_float);

  cl::Buffer d_x = cl::Buffer{context, CL_MEM_ALLOC_HOST_PTR, buffer_size};
  cl::Buffer d_y = cl::Buffer{context, CL_MEM_ALLOC_HOST_PTR, buffer_size};
  cl::Buffer d_z = cl::Buffer{context, CL_MEM_ALLOC_HOST_PTR, buffer_size};

  float *x = (float *)queue.enqueueMapBuffer(d_x, CL_TRUE, CL_MAP_WRITE, 0,
                                             buffer_size);
  float *y = (float *)queue.enqueueMapBuffer(d_y, CL_TRUE, CL_MAP_WRITE, 0,
                                             buffer_size);
  float *z = (float *)queue.enqueueMapBuffer(d_z, CL_TRUE, CL_MAP_WRITE, 0,
                                             buffer_size);

  for (size_t idx = 0; idx < num_elements; idx++) {
    x[idx] = 1.0f;
    y[idx] = 2.0f;
    z[idx] = 0.0f;
  }

  queue.enqueueUnmapMemObject(d_x, x);
  queue.enqueueUnmapMemObject(d_y, y);
  queue.enqueueUnmapMemObject(d_z, z);

  kernel.setArg(0, 2.0f);
  kernel.setArg(1, d_x);
  kernel.setArg(2, d_y);
  kernel.setArg(3, d_z);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{num_elements});

  z = (float *)queue.enqueueMapBuffer(d_z, CL_TRUE, CL_MAP_READ, 0,
                                      buffer_size);

  float error = 0.0;
  for (size_t idx = 0; idx < num_elements; idx++) {
    error = fmax(error, fabs(z[idx] - 4.0f));
  }
  printf("error: %e (%s)\n", error, error == 0.0 ? "PASS" : "FAIL");

  queue.enqueueUnmapMemObject(d_z, z);

  return 0;
}
