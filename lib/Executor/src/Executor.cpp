#include "Executor.h"
#include <algorithm>
#include <cassert>

float getRuntimeInMilliseconds(cl::Event event)
{
  cl_ulong start;
  cl_ulong end;
  cl_int err;

  event.wait();

  err = clGetEventProfilingInfo(event(), CL_PROFILING_COMMAND_START,
                                sizeof(start), &start, NULL);
  ASSERT(err == CL_SUCCESS);

  err = clGetEventProfilingInfo(event(), CL_PROFILING_COMMAND_END,
                                sizeof(end), &end, NULL);
  ASSERT(err == CL_SUCCESS);

  return static_cast<float> ((end - start) * 1.0e-06);
}

float getRuntimeInMilliseconds(cl::Event &start, cl::Event &end) {
  cl_ulong time_start, time_end;
  cl_int err;

  end.wait();
  
  err = clGetEventProfilingInfo(start(), CL_PROFILING_COMMAND_SUBMIT, sizeof(start), &time_start, NULL);

  ASSERT(err == CL_SUCCESS);

  err = clGetEventProfilingInfo(end(), CL_PROFILING_COMMAND_SUBMIT, sizeof(end), &time_end, NULL);

  ASSERT(err == CL_SUCCESS);

  return static_cast<float> ((time_end - time_start) * 1.0e-06);
}

void initExecutor(int platformId, int deviceId)
{
  executor::init(executor::platform(platformId), executor::device(deviceId));
}

void initExecutor(std::string deviceTypeString)
{
  executor::device_type deviceType;
  std::istringstream(deviceTypeString) >> deviceType;
  executor::init(executor::nDevices(1).deviceType(deviceType));
}

void shutdownExecutor()
{
  executor::terminate();
}

std::string getPlatformName()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->clPlatform().getInfo<CL_PLATFORM_NAME>();
}

unsigned long getDeviceLocalMemSize()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->localMemSize();
}

unsigned long getDeviceGlobalMemSize()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->globalMemSize();
}

unsigned long getDeviceMaxMemAllocSize()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->maxMemAllocSize();
}

unsigned long getDeviceMaxWorkGroupSize()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->maxWorkGroupSize();
}

std::string getDeviceName()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->name();
}

std::string getDeviceType()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->typeAsString();
}

bool supportsDouble()
{
  auto& devicePtr = executor::globalDeviceList.front();
  return devicePtr->supportsDouble();
}

executor::KernelTime executeKernel(cl::Kernel kernel,
                     int localSize1, int localSize2, int localSize3,
                     int globalSize1, int globalSize2, int globalSize3,
                     const std::vector<executor::KernelArg*>& args)
{
  executor::KernelTime time;
  cl::Event totalBeginn, uploadBeginn, uploadEnd, downloadBeginn, downloadEnd, totalEnd;

  auto& devPtr = executor::globalDeviceList.front();

  devPtr->enqueueMarker(&totalBeginn);

  cl_uint clLocalSize1 = localSize1;
  cl_uint clGlobalSize1 = globalSize1;
  cl_uint clLocalSize2 = localSize2;
  cl_uint clGlobalSize2 = globalSize2;
  cl_uint clLocalSize3 = localSize3;
  cl_uint clGlobalSize3 = globalSize3;

  devPtr->enqueueMarker(&uploadBeginn);

  int i = 0;
  for (auto& arg : args) {
    arg->upload();
    arg->setAsKernelArg(kernel, i);
    ++i;
  }

  devPtr->enqueueMarker(&uploadEnd);
  time.upload = getRuntimeInMilliseconds(uploadBeginn, uploadEnd);

  auto event = devPtr->enqueue(kernel,
                               cl::NDRange(clGlobalSize1,
                                           clGlobalSize2, clGlobalSize3),
                               cl::NDRange(clLocalSize1,
                                           clLocalSize2, clLocalSize3));
  time.launch = getRuntimeInMilliseconds(event);

  devPtr->enqueueMarker(&downloadBeginn);

  for (auto& arg : args) arg->download();

  devPtr->enqueueMarker(&downloadEnd);

  time.download = getRuntimeInMilliseconds(downloadBeginn, downloadEnd);

  devPtr->enqueueMarker(&totalEnd);
  time.total = getRuntimeInMilliseconds(totalBeginn, totalEnd);

  return time;
}

executor::KernelTime execute(const executor::Kernel& kernel,
               int localSize1, int localSize2, int localSize3,
               int globalSize1, int globalSize2, int globalSize3,
               const std::vector<executor::KernelArg*>& args)
{
  return executeKernel(kernel.build(), localSize1, localSize2, localSize3,
                       globalSize1, globalSize2, globalSize3, args);
}

void benchmark(const executor::Kernel& kernel,
               int localSize1, int localSize2, int localSize3,
               int globalSize1, int globalSize2, int globalSize3,
               const std::vector<executor::KernelArg*>& args,
               int iterations, double timeout,
               std::vector<executor::KernelTime>& runtimes)
{
  for (int i = 0; i < iterations; i++) {
    //std::cout << "Iteration: " << i << '\n';

    for(auto& arg:args){
      arg->clear();
    }

    executor::KernelTime runtime = executeKernel(kernel.build(), localSize1, localSize2, localSize3,
                       globalSize1, globalSize2, globalSize3, args);

    runtimes.push_back(runtime);
    
    if (timeout != 0.0 && runtime.launch >= timeout) {
      return;
    }
  }
}

double evaluate(const executor::Kernel& kernel,
                int localSize1, int localSize2, int localSize3,
                int globalSize1, int globalSize2, int globalSize3,
                const std::vector<executor::KernelArg*>& args,
                int iterations, double timeout)
{
  auto& devPtr = executor::globalDeviceList.front();
  cl_uint clLocalSize1 = localSize1;
  cl_uint clGlobalSize1 = globalSize1;
  cl_uint clLocalSize2 = localSize2;
  cl_uint clGlobalSize2 = globalSize2;
  cl_uint clLocalSize3 = localSize3;
  cl_uint clGlobalSize3 = globalSize3;

  cl_int err = CL_SUCCESS;

  // Copy the buffers only once
  for (auto& arg : args) {
    arg->upload();
  }

  { // run a single workgroup on dummy data
    auto k = executor::Kernel(
        std::string{"#define WORKGROUP_GUARD {for(int i = 0; i < get_work_dim(); ++i) if(get_group_id(i)!=0) return;}\n"} + kernel.getSource(),
           kernel.getName(), kernel.getBuildOptions()
        );

    auto openclKernel = k.build();

    auto wg_size = -1;
    err = openclKernel.getWorkGroupInfo(devPtr->clDevice(), CL_KERNEL_WORK_GROUP_SIZE ,&wg_size);
    if(err != CL_SUCCESS) {
      std::cerr << "ERROR " << err << std::endl;
      return -1;
    }

    cl_ulong private_mem = -1;
    err = openclKernel.getWorkGroupInfo(devPtr->clDevice(), CL_KERNEL_PRIVATE_MEM_SIZE, &private_mem);
    if(err != CL_SUCCESS) {
      std::cerr << "ERROR " << err << std::endl;
      return -1;
    }

   std::cout << "Amount of private memory: " << private_mem << std::endl;

    

    if(wg_size < localSize1 * localSize2 * localSize3) {
      return -1;
    }
    
    int i = 0;
    for (auto& arg : args) {
      arg->setAsKernelArg(openclKernel, i);
      ++i;
    }

    auto event = devPtr->enqueue(openclKernel,
                                 cl::NDRange(clGlobalSize1,
                                             clGlobalSize2, clGlobalSize3),
                                 cl::NDRange(clLocalSize1,
                                             clLocalSize2, clLocalSize3));

    auto time = getRuntimeInMilliseconds(event);
    std::cout << "Single workgroup: " << time << std::endl;
    if (time > timeout) return -time;
  }


  { // actual run
    std::vector<double> allRuntimes;
    allRuntimes.resize(iterations);

    auto openclKernel = kernel.build();
    int i = 0;
    for (auto& arg : args) {
      arg->setAsKernelArg(openclKernel, i);
      ++i;
    }

    for (int i = 0; i < iterations; i++) {
      auto event = devPtr->enqueue(openclKernel,
                                   cl::NDRange(clGlobalSize1,
                                               clGlobalSize2, clGlobalSize3),
                                   cl::NDRange(clLocalSize1,
                                               clLocalSize2, clLocalSize3));
      auto runtime = getRuntimeInMilliseconds(event);
      if(runtime > timeout) {
        for (auto& arg : args) arg->download();
        return runtime;
      }
      allRuntimes[i] = runtime;
    }
  
    for (auto& arg : args) arg->download();

    std::sort(std::begin(allRuntimes), std::end(allRuntimes));

    double median;

    if (iterations % 2 == 0)
      median = (allRuntimes[iterations/2] + allRuntimes[iterations/2 - 1]) / 2.0;
    else 
      median = allRuntimes[iterations/2];

    return median;
  }
}

