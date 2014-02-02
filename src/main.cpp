#include <CL/cl.hpp>
#include <chrono>

#include <initializer_list>
#include <array>

#include <iostream>
#include "openCLUtilities.hpp"

using namespace std;


int main()
{

    //get all platforms (drivers)
    vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0)
    {
        cout << "No platforms found. Check OpenCL installation!" << endl;
        exit(1);
    }

    for (cl::Platform p : all_platforms)
    {
        cout << "Available platform: " << p.getInfo<CL_PLATFORM_NAME>() << endl;
    }

    cl::Platform default_platform = all_platforms[0];
    cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

    //get default device of the default platform
    vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0)
    {
        cout << " No devices found. Check OpenCL installation!" << endl;
        exit(1);
    }
    for (cl::Device d : all_devices)
    {
        cout << "Available platform: " << d.getInfo<CL_DEVICE_NAME>() << endl;
    }
    cl::Device default_device = all_devices[0];
    cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";


    cl::Context context({default_device});

    cl::Program program = buildProgramFromSource(context, "../src/perceptron_layer.cl");


    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    const int in_layer_size = 2;
    const int out_layer_size = 3;
    cl_float in_values[in_layer_size] { 1., 2. };
    cl_float in_weights[in_layer_size][out_layer_size] {{1., 2., 3.},
                                                        {4., 5., 6.}};
    cl_float out_values[out_layer_size] {0.};

    // Creates buffer on the device
    cl::Buffer buf_out_layer_size(context, CL_MEM_READ_ONLY, sizeof(cl_int));
    cl::Buffer buf_in_layer_size(context, CL_MEM_READ_ONLY, sizeof(cl_int));
    cl::Buffer buf_in_values(context, CL_MEM_READ_ONLY, sizeof(cl_float) * in_layer_size);
    cl::Buffer buf_in_weights(context, CL_MEM_READ_ONLY, sizeof(cl_float) * in_layer_size * out_layer_size);
    cl::Buffer buf_out_values(context, CL_MEM_READ_WRITE, sizeof(cl_float) * out_layer_size);

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, default_device);

    // Prepare device memory for each layer
    if(queue.enqueueWriteBuffer(buf_in_values, CL_TRUE, 0, sizeof(cl_float)*in_layer_size, in_values) != CL_SUCCESS)
    {
        cerr << "Error while pushing data to the device" <<  endl;
        exit(0);
    }
    if(queue.enqueueWriteBuffer(buf_in_weights, CL_TRUE, 0, sizeof(cl_float)*out_layer_size*in_layer_size, in_weights) != CL_SUCCESS) {
        cerr << "Error while pushing data to the device" <<  endl;
        exit(0);
    }
    if(queue.enqueueWriteBuffer(buf_out_layer_size, CL_TRUE, 0, sizeof(cl_int), &out_layer_size) != CL_SUCCESS) {
        cerr << "Error while pushing data to the device" <<  endl;
        exit(0);
    }
    if(queue.enqueueWriteBuffer(buf_in_layer_size, CL_TRUE, 0, sizeof(cl_int), &in_layer_size) != CL_SUCCESS) {
        cerr << "Error while pushing data to the device" <<  endl;
        exit(0);
    }

    //run the kernel
    cl::KernelFunctor perceptron_kernel(cl::Kernel(program, "perceptron"), queue, cl::NullRange, cl::NDRange(out_layer_size), cl::NullRange);
    perceptron_kernel(buf_in_layer_size, buf_out_layer_size, buf_in_values, buf_in_weights, buf_out_values);

    //alternative way to run the kernel
    /*cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
    queue.finish();*/

    //read result C from the device to array C
    queue.enqueueReadBuffer(buf_out_values, CL_TRUE, 0, sizeof(cl_float)*out_layer_size, out_values);
    end = std::chrono::system_clock::now();

    cout << "Out values for layer1: ";
    for(const auto& i : out_values) {
        cout << i << "\t";
    }
    cout << endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "\n\nfinished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
