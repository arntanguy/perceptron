#include <CL/cl.hpp>
#include <chrono>

#include <list>

#include "perceptron.hpp"


using namespace std;


int main(int argc, char **argv)
{
    std::srand(std::time(0)); // use current time as seed for random generator

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

    cl::Platform default_platform;
    default_platform = all_platforms[0];
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

    //NeuronLayer<cl_float> out_layer(1000, nullptr);
    //NeuronLayer<cl_float> hidden_layer(10000, &out_layer);
    //NeuronLayer<cl_float> input_layer(1000, &hidden_layer);

    //create queue to which we will push commands for the device.
    cl::Kernel perceptronKernel(program, "perceptron");
    cl::CommandQueue queue(context, default_device);


    //NeuronLayer<cl_float> input_layer(2);
    //NeuronLayer<cl_float> hidden_layer(3);
    //NeuronLayer<cl_float> out_layer(1);
    ////input_layer.setOutputLayer(&hidden_layer);
    ////hidden_layer.setOutputLayer(&out_layer);
    ////out_layer.setOutputLayer(nullptr);

    ////input_layer.initRandomWeights();
    //
    //input_layer.setValues({ 1., 2.});
    //input_layer.setWeights({1., 2., 3.,
    //        4., 5., 6.});
    //hidden_layer.setValues({0., 1., 2.});
    //out_layer.setValues({1.});
    //hidden_layer.setWeights({1., 1., 1.});
    ////cout << "In Layer: \n" << input_layer << endl;
    ////cout << "\nHidden Layer:\n " << hidden_layer << endl;
    ////cout << "\nOut Layer: \n" << out_layer << endl;

    //input_layer.createBuffers(context);
    //hidden_layer.createBuffers(context);
    //out_layer.createBuffers(context);

    ////input_layer.enqueueWriteBuffers(queue);
    ////hidden_layer.enqueueWriteBuffers(queue);
    ////out_layer.enqueueWriteBuffers(queue);

    Perceptron<cl_float> perceptron(context, queue);
    //// Creates the layers, reserve data on GPU
    perceptron.createLayer(2);
    perceptron.createLayer(3);
    perceptron.createLayer(1);

    // Define weights between layers
    std::list<std::list<cl_float>> weights = {{1., 2., 3., 4., 5., 6.}, // between input layer and hidden_layer1
                                              {1., 2., 3.}}; // between hidden layer 1 and out layer
    perceptron.setWeights(weights);
    perceptron.setInputValues({1., 2.});

    //// Upload all of the data on the GPU
    perceptron.upload();
    perceptron.displayAll();
    //// Run the kernel 
    perceptron.run(perceptronKernel);


    //input_layer.enqueueRun(perceptronKernel, queue);
    //hidden_layer.enqueueRun(perceptronKernel, queue);

    //input_layer.enqueueReadBuffers(queue);
    //cout << "\nInput Layer(computed): " << input_layer << endl;
    //hidden_layer.enqueueReadBuffers(queue);
    //cout << "\nHidden Layer(computed): " << hidden_layer << endl;
    //out_layer.enqueueReadBuffers(queue);
    //cout << "\nOut Layer(computed): \n" << out_layer << endl;

    end = std::chrono::system_clock::now();


    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "\n\nfinished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
