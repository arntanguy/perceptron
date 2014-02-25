#include <CL/cl.hpp>
#include <chrono>

#include <list>

#include "perceptron.hpp"


using namespace std;


int main(int argc, char **argv)
{
    std::srand(std::time(0)); // use current time as seed for random generator

    //get all platforms (drivers)
    cout << "=========" << endl;
    cout << "Platform" << endl;
    cout << "=========" << endl;
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
    cout << endl << endl;

    cl::Context context({default_device});

    cl::Program program = buildProgramFromSource(context, "../src/perceptron_layer.cl");

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    cout << "=====================" << endl;
    cout << "Setting up perceptron" << endl;
    cout << "=====================" << endl;

    //create queue to which we will push commands for the device.
    cl::Kernel perceptronKernel(program, "perceptron");
    // Training
    cl::Kernel perceptronTrainOutputKernel(program, "perceptron_train_output_layer");
    cl::Kernel perceptronTrainBackpropagate(program, "perceptron_train_backpropagate");
    cl::Kernel perceptronTrainUpdateWeights(program, "perceptron_train_update_weights");

    cl::CommandQueue queue(context, default_device);

    Perceptron<cl_float> perceptron(context, queue);
    // Creates the layers, reserve data on GPU
    perceptron.createLayer(2);
    perceptron.createLayer(2);
    perceptron.createLayer(2);
    perceptron.createLayer(1);
    //perceptron.createLayer(100000);
    //perceptron.createLayer(10000);
    //perceptron.createLayer(316);
    //perceptron.createLayer(17);

    // Define weights between layers
    //std::list<std::list<cl_float>> weights = {{.1, .2, .3, .4, .5, .06}, // between input layer and hidden_layer1
    //                                          {.1, .2, .3}}; // between hidden layer 1 and out layer
    //std::list<std::list<cl_float>> weights = {{1, 2, 3, 4, 5, 6}, // between input layer and hidden_layer1
    //                                          {1, 2, 3}}; // between hidden layer 1 and out layer
    //perceptron.setWeights(weights);
    perceptron.setInputValues({0., 0.});

    //// Upload all of the data on the GPU
    cout << "Uploading perceptron data to the GPU" << endl;
    perceptron.upload();
    


    cout << "=====================" << endl;
    cout << "Training Perceptron" << endl; 
    cout << "=====================" << endl;
    perceptron.setWeights( {{0.25, -0.25, 0.25, -0.35, 0.25, 0.25},
                            {0.25, -0.35, -0.35, 0.15, -0.25, 0.15},
                            {0.5, 0.5, 0.35}} );
    //perceptron.setWeights( {{0.25, -0.35, 0.25, -0.25, 0.25,  0.25} ,
    //                        {0.25, 0.15, -0.35,  -0.35, -0.25, 0.15},
    //                        {0.5, 0.5, 0.35}});
                            
    //perceptron.initRandomWeights();
    cout << endl;
    cout << "Perceptron before training" << endl;
    perceptron.run(perceptronKernel);
    perceptron.enqueueReadAllBuffers();
    perceptron.displayAll();
    cout << endl;

    perceptron.train(perceptronKernel, perceptronTrainOutputKernel,
                     perceptronTrainBackpropagate, perceptronTrainUpdateWeights,
                     {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}},
                     {{0.}, {1.}, {1.}, {0.}});

    cout << endl;
    cout << "After training: " << endl;
    perceptron.enqueueReadAllBuffers();
    perceptron.displayAll();


    //// Run the kernel 
    cout << endl;
    cout << "=====================" << endl;
    cout << "Running  perceptron" << endl;
    cout << "=====================" << endl;
    cout << "Running xor(1, 0)" << endl;
    perceptron.setInputValues({1,0});
    perceptron.run(perceptronKernel);
    perceptron.enqueueReadAllBuffers();
    perceptron.displayAll();

    cout << "Running xor(0, 1)" << endl;
    perceptron.setInputValues({0,1});
    perceptron.run(perceptronKernel);
    perceptron.enqueueReadAllBuffers();
    perceptron.displayAll();

    cout << "Running xor(1, 1)" << endl;
    perceptron.setInputValues({1,1});
    perceptron.run(perceptronKernel);
    perceptron.enqueueReadAllBuffers();
    perceptron.displayAll();

    cout << "Running xor(0, 0)" << endl;
    perceptron.setInputValues({0,0});
    perceptron.run(perceptronKernel);
    perceptron.enqueueReadAllBuffers();
    perceptron.displayAll();

    cout << endl;
    cout << "===================" << endl;
    cout  << "Final result: " << endl;
    cout << "===================" << endl;
    perceptron.enqueueReadAllBuffers();
    perceptron.displayAll();


    end = std::chrono::system_clock::now();


    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "\n\nfinished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
