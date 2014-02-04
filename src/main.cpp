#include <CL/cl.hpp>
#include <chrono>

#include <initializer_list>
#include <array>

#include <iostream>
#include "openCLUtilities.hpp"

using namespace std;


/**
 * @brief NeuronLayer represents one of the perceptron neuron layers.
 * Due to GPU limitation regarding dynamic pointers within structures, it is
 * strucured to closely match the opencl kernel implementation.
 * The data structures are composed of
 * - an array of values, each value representing the internal value of one
 *   neuron in the layer
 * - An array of weights. A "line" i in the array represents the weights for
 *   the neuron i.
 * See the opencl Kernel implementation for more details
 */
template<typename T>
class NeuronLayer
{
 private:
  cl::Buffer buf_in_size;
  cl::Buffer buf_in_values;
  cl::Buffer buf_in_weights;

  const cl_int m_in_size;
  cl_int m_out_size = 0;

  // Linked list
  NeuronLayer* m_out_layer;

 public:
  T* values = nullptr;
  // Weights to the next layer
  T* weights = nullptr;

  NeuronLayer(const cl_int& in_s, NeuronLayer *out_layer) : m_in_size(in_s), m_out_layer(out_layer) {
      values = new T[m_in_size];
      if(out_layer != nullptr) {
          const cl_int& out_size = out_layer->getInSize();
          weights = new T[out_size];
          m_out_size = out_size;
      } else {
          m_out_size = 0;
          cout << "NULL PTR: " << m_out_size << endl;
      }
  }

  cl_int getInSize() const {
      return m_in_size;
  }

  void setValues(std::initializer_list<T> init) {
      if(init.size() != m_in_size) {
          throw "Your initializer list for values exeeds the maximum size!";
      }

      int j=0;
      for(const auto& i: init) {
          values[j++] = i;
      }
  }

  void setWeights(std::initializer_list<T> init) {
      cout << "size: " << m_in_size <<"," << m_out_size <<", " << m_in_size * m_out_size << ", init: " << init.size() << endl;
      if(m_out_layer != nullptr && init.size() != m_in_size * m_out_size) {
          cout << "in: " << m_in_size << ", out: " << m_out_size << endl;
          throw "Your initializer list for weights exeeds the maximum size!";
      }

      auto it = init.begin();
      int j=0;
      while(it != end(init) && j < m_in_size * m_out_size)
      {
          cout << *it << endl;
          weights[j++] = *it;
          ++it;
      }
  }


  void display() const {
      cout << "\tValues: ";
      for(int i=0; i<m_in_size; i++) {
          cout  << values[i] << "\t" ;
      }
      cout << "\n\tWeights: ";
      if(m_out_layer != nullptr) {
        for(int i=0; i<m_in_size*m_out_size; i++) {
            cout << weights[i] << "\t" ;
        }
      } else {
          cout << "\tNo weights defined" << endl;
      }
  }

  /**
   * @brief Prepares the buffer (host size)
   */
  void createBuffers(cl::Context& context)
  {
      // Creates buffer on the device
      buf_in_size = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_int));
      buf_in_values = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * m_in_size);
      buf_in_weights = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * m_in_size * m_out_size);
  }

  void enqueueWriteBuffers(cl::CommandQueue& queue)
  {
      // Prepare device memory for each layer
      queue.enqueueWriteBuffer(buf_in_size, CL_TRUE, 0, sizeof(T), &m_in_size);
      queue.enqueueWriteBuffer(buf_in_values, CL_TRUE, 0, sizeof(T)*m_in_size, values);
      queue.enqueueWriteBuffer(buf_in_weights, CL_TRUE, 0, sizeof(T)*m_out_size*m_in_size, weights);
  }

  void enqueueReadBuffers(cl::CommandQueue& queue)
  {
      queue.enqueueReadBuffer(buf_in_values, CL_TRUE, 0, sizeof(T)*m_in_size, values);
  }

  // Should only be called by run (existence of last element not checked)
  cl::Buffer getValuesBuf() const {
      return buf_in_values;
  }
  cl::Buffer getLayerSizeBuf() const {
      return buf_in_size;
  }

  void run(cl::Kernel &kernel, cl::CommandQueue& queue) {
      if(m_out_layer != nullptr) {
        kernel.setArg(0, buf_in_size);
        kernel.setArg(1, m_out_layer->getLayerSizeBuf());
        kernel.setArg(2, buf_in_values);
        kernel.setArg(3, buf_in_weights);
        kernel.setArg(4, m_out_layer->getValuesBuf());
        cout << "Setting up kernel with ND-range " << m_out_size << endl;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,cl::NDRange(m_out_size),cl::NullRange);
        queue.finish();
      } else {
        throw "Error: no layers left!";
      }
  }

};


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

    NeuronLayer<cl_float> out_layer(1, nullptr);
    NeuronLayer<cl_float> hidden_layer(3, &out_layer);
    NeuronLayer<cl_float> input_layer(2, &hidden_layer);

    hidden_layer.setValues({0., 0., 0.});
    hidden_layer.setWeights({1., 2., 3.});
    input_layer.setValues({ 1., 2.});
    input_layer.setWeights({1., 2., 3.,
                           4., 5., 6.});
    cout << "In Layer: \n";
    input_layer.display();
    cout << "\nHidden Layer:\n ";
    hidden_layer.display();
    cout << "\nOut Layer: \n";
    out_layer.display();

    //create queue to which we will push commands for the device.
    cl::Kernel perceptronKernel(program, "perceptron");
    cl::CommandQueue queue(context, default_device);
    input_layer.createBuffers(context);
    hidden_layer.createBuffers(context);
    out_layer.createBuffers(context);
    input_layer.enqueueWriteBuffers(queue);
    hidden_layer.enqueueWriteBuffers(queue);
    out_layer.enqueueWriteBuffers(queue);

    input_layer.run(perceptronKernel, queue);
    hidden_layer.run(perceptronKernel, queue);

    hidden_layer.enqueueReadBuffers(queue);
    cout << "\nHidden Layer(computed): " << endl;
    hidden_layer.display();

    out_layer.enqueueReadBuffers(queue);
    cout << "\nOut Layer(computed): " << endl;
    out_layer.display();


    //run the kernel
    //cl::KernelFunctor perceptron_kernel(cl::Kernel(program, "perceptron"), queue, cl::NullRange, cl::NDRange(out_layer_size), cl::NullRange);
    //perceptron_kernel(buf_in_layer_size, buf_out_layer_size, buf_in_values, buf_in_weights, buf_out_values);

    //read result C from the device to array C
    //queue.enqueueReadBuffer(buf_out_values, CL_TRUE, 0, sizeof(cl_float)*out_layer_size, out_values);
    //end = std::chrono::system_clock::now();

    //cout << "Out values for layer1: ";
    //for(const auto& i : out_values) {
    //    cout << i << "\t";
    //}
    //cout << endl;

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "\n\nfinished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
