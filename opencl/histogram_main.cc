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

#define NUM_BUCKETS 128

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

  ret.resize(filesize);
  ret.insert(ret.begin(), std::istreambuf_iterator<char>(is),
             std::istreambuf_iterator<char>());

  return ret;
}

static cl::Program createProgramWithIL(const cl::Context &context,
                                       const std::vector<cl_uchar> &il) {
  return cl::Program{clCreateProgramWithIL(context(), il.data(), il.size(), nullptr)};
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
            "Usage: histogram [options]\n"
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

  cl::Context context{devices[device_index]};
  cl::CommandQueue queue = cl::CommandQueue{context, devices[device_index]};

  const char *filename =
      (sizeof(void *) == 8) ? "histogram_kernel64.spv" : "histogram_kernel32.spv";
  std::vector<cl_uchar> spirv = read_file(filename);

  cl::Program program = createProgramWithIL(context, spirv);
  const char *build_options = NULL;
  program.build(build_options);

  for (auto &device : program.getInfo<CL_PROGRAM_DEVICES>()) {
    printf("Program build log for device %s:\n",
           device.getInfo<CL_DEVICE_NAME>().c_str());
    printf("%s\n", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device).c_str());
  }

  const char *kernel_name = "histogram";
  cl::Kernel kernel = cl::Kernel{program, kernel_name};

  size_t num_elements = 1 << 20;
  size_t data_size = num_elements * sizeof(cl_float);
  size_t histogram_size = NUM_BUCKETS * sizeof(cl_uint);

  cl::Buffer d_data = cl::Buffer{context, CL_MEM_ALLOC_HOST_PTR, data_size};
  cl::Buffer d_histogram = cl::Buffer{context, CL_MEM_ALLOC_HOST_PTR, histogram_size};

  float *data = (float *)queue.enqueueMapBuffer(d_data, CL_TRUE, CL_MAP_WRITE, 0,
						data_size);
  float *histogram = (float *)queue.enqueueMapBuffer(d_histogram, CL_TRUE, CL_MAP_WRITE, 0,
						     histogram_size);

  float range = (float)RAND_MAX;
  for (size_t idx = 0; idx < num_elements; idx++) {
    data[idx] = rand();
  }
  for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
    histogram[idx] = 0;
  }

  queue.enqueueUnmapMemObject(d_data, data);
  queue.enqueueUnmapMemObject(d_histogram, histogram);

  kernel.setArg(0, num_elements);
  kernel.setArg(1, range);
  kernel.setArg(2, d_data);
  kernel.setArg(3, d_histogram);

  size_t elts_per_thread = 16;
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{num_elements/elts_per_thread});

  histogram = (float *)queue.enqueueMapBuffer(d_histogram, CL_TRUE, CL_MAP_READ, 0,
					      histogram_size);

  size_t total = 0;
  for (size_t idx = 0; idx < NUM_BUCKETS; idx++) {
    total += histogram[idx];
    printf("histogram[%lu] = %u\n", idx, histogram[idx]);
  }
  printf("\ntotal = %lu (%s)\n", total,
         total == num_elements ? "PASS" : "FAIL");

  queue.enqueueUnmapMemObject(d_histogram, histogram);

  return 0;
}
