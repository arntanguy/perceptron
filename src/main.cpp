#include <iostream>
#include <CL/cl.hpp>

#include <fstream>
#include <string>

#include <cstdlib>
#include <ctime>

#include <initializer_list>
#include <array>

#include "openCLUtilities.hpp"

using namespace std;


template<typename T, int size>
struct Neuron
{
    T value = 0;
    typedef std::array<T, size> weights_t;
    weights_t weights;

    //T weights[size];

 public:
    template<typename... E>
    Neuron(E... ts) : weights{ts...} { }
    Neuron() {}
    void display() const {
        cout << "Value: " << value << endl;
        cout << "Weights: ";
        for(auto i : weights) {
            cout << i << "\t" << endl;
        }
    }
};

int main()
{

    /* initialize random seed: */
    srand (time(NULL));

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

    cl_float A[10] = {0};
    cl_float B[10] = {0};
    cl_float C[10] = {0};

    // Input layer
    // Weights between input layer and layer1
    //Neuron<cl_float, 3> input_neuron({ 0.f, 1.f, 2.f });
    //input_neuron.value = 42;
    //Neuron<cl_float, 3> input_layer[1] { input_neuron };

    //// Hidden layer
    //Neuron<cl_float, 1> layer1[3] { Neuron<cl_float, 1>{1.f} , Neuron<cl_float, 1>{2.f}, Neuron<cl_float, 1>{3.f} };
    //cout << "Host: Displaying Layer1: " << endl;
    //for(auto i : layer1) {
    //    cout << "Host: Neurone layer1: ";
    //    i.display();
    //}

    //// Output layer
    //Neuron<cl_float, 1> output_layer[1];

    const int previous_size = 2;
    const int current_size = 3;
    cl_float in_values[previous_size] { 1., 2. };
    cl_float in_weights[2][3] {{1., 2., 3.},
                               {4., 5., 6.}};
    cl_float out_values[3] {0.};

    // Creates buffer on the device
    cl::Buffer buf_current_layer_size(context, CL_MEM_READ_ONLY, sizeof(cl_int));
    cl::Buffer buf_previous_layer_size(context, CL_MEM_READ_ONLY, sizeof(cl_int));
    cl::Buffer buf_in_values(context, CL_MEM_READ_ONLY, sizeof(cl_float) * previous_size);
    cl::Buffer buf_in_weights(context, CL_MEM_READ_ONLY, sizeof(cl_float) * previous_size * current_size);
    cl::Buffer buf_out_values(context, CL_MEM_READ_WRITE, sizeof(cl_float) * current_size);

    //create queue to which we will push commands for the device.
    cl::CommandQueue queue(context, default_device);

    // Prepare device memory for each layer
    if(queue.enqueueWriteBuffer(buf_in_values, CL_TRUE, 0, sizeof(cl_float)*previous_size, in_values) != CL_SUCCESS)
    {
        cerr << "Error" << endl;
        exit(0);
    }
    if(queue.enqueueWriteBuffer(buf_in_weights, CL_TRUE, 0, sizeof(cl_float)*current_size*previous_size, in_weights) != CL_SUCCESS) {
        cerr << "Error" << endl;
        exit(0);
    }
    if(queue.enqueueWriteBuffer(buf_current_layer_size, CL_TRUE, 0, sizeof(cl_int), &current_size) != CL_SUCCESS) {
        cerr << "Error" << endl;
        exit(0);
    }
    if(queue.enqueueWriteBuffer(buf_previous_layer_size, CL_TRUE, 0, sizeof(cl_int), &previous_size) != CL_SUCCESS) {
        cerr << "Error" << endl;
        exit(0);
    }

    //run the kernel
    cl::KernelFunctor perceptron_kernel(cl::Kernel(program, "perceptron"), queue, cl::NullRange, cl::NDRange(3), cl::NullRange);
    perceptron_kernel(buf_current_layer_size, buf_previous_layer_size, buf_in_values, buf_in_weights, buf_out_values);

    //alternative way to run the kernel
    /*cl::Kernel kernel_add=cl::Kernel(program,"simple_add");
    kernel_add.setArg(0,buffer_A);
    kernel_add.setArg(1,buffer_B);
    kernel_add.setArg(2,buffer_C);
    queue.enqueueNDRangeKernel(kernel_add,cl::NullRange,cl::NDRange(10),cl::NullRange);
    queue.finish();*/

    //read result C from the device to array C
    queue.enqueueReadBuffer(buf_out_values, CL_TRUE, 0, sizeof(cl_float)*current_size, out_values);
    cout << "Out values for layer1: ";
    for(const auto& i : out_values) {
        cout << i << "\t";
    }
    cout << endl;

    return 0;
}
