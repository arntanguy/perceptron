#include <CL/cl.hpp>
#include <chrono>

#include <initializer_list>
#include <array>

#include <iostream>
#include <sstream>
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
  cl::Buffer buf_size;
  cl::Buffer buf_values;
  cl::Buffer buf_weights;

  const cl_int m_size;
  cl_int m_out_size = 0;

  // Linked list
  NeuronLayer* m_out_layer;

 public:
  T* values = nullptr;
  // Weights to the next layer
  T* weights = nullptr;

  NeuronLayer(const cl_int& in_s, NeuronLayer *out_layer) : m_size(in_s), m_out_layer(out_layer) {
      values = new T[m_size];
      if(out_layer != nullptr) {
          const cl_int& out_size = out_layer->getInSize();
          weights = new T[m_size*out_size];
          m_out_size = out_size;
      } else {
          m_out_size = 0;
          cout << "NULL PTR: " << m_out_size << endl;
      }
  }

  cl_int getInSize() const {
      return m_size;
  }

  void setValues(std::initializer_list<T> init) {
      if(init.size() != m_size) {
          throw "Your initializer list for values exeeds the maximum size!";
      }

      int j=0;
      for(const auto& i: init) {
          values[j++] = i;
      }
  }

  void setWeights(std::initializer_list<T> init) {
      if(m_out_layer != nullptr && init.size() != m_size * m_out_size) {
          throw "Your initializer list for weights exeeds the maximum size!";
      }

      int j=0;
      for(const auto& val : init)
      {
          weights[j++] = val;
      }
  }

  /**
   * @brief Prepares the buffer (host size)
   */
  void createBuffers(cl::Context& context)
  {
      // Creates buffer on the device
      buf_size = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_int));
      buf_values = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * m_size);
      buf_weights = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * m_size * m_out_size);
  }

  void enqueueWriteBuffers(cl::CommandQueue& queue)
  {
      // Prepare device memory for each layer
      queue.enqueueWriteBuffer(buf_size, CL_TRUE, 0, sizeof(T), &m_size);
      queue.enqueueWriteBuffer(buf_values, CL_TRUE, 0, sizeof(T)*m_size, values);
      queue.enqueueWriteBuffer(buf_weights, CL_TRUE, 0, sizeof(T)*m_out_size*m_size, weights);
  }

  void enqueueReadBuffers(cl::CommandQueue& queue)
  {
      queue.enqueueReadBuffer(buf_values, CL_TRUE, 0, sizeof(T)*m_size, values);
  }

  // Should only be called by run (existence of last element not checked)
  cl::Buffer getValuesBuf() const {
      return buf_values;
  }
  cl::Buffer getLayerSizeBuf() const {
      return buf_size;
  }

  void run(cl::Kernel &kernel, cl::CommandQueue& queue) {
      if(m_out_layer != nullptr) {
        kernel.setArg(0, buf_size);
        kernel.setArg(1, m_out_layer->getLayerSizeBuf());
        kernel.setArg(2, buf_values);
        kernel.setArg(3, buf_weights);
        kernel.setArg(4, m_out_layer->getValuesBuf());
        cout << "Setting up kernel with ND-range " << m_out_size << endl;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange,cl::NDRange(m_out_size),cl::NullRange);
        queue.finish();
      } else {
        throw "Error: no layers left!";
      }
  }

    friend ostream& operator<< (ostream &out, const NeuronLayer& layer) {
      out << "\tValues: ";
      for(int i=0; i<layer.m_size; i++) {
          out  << layer.values[i] << "\t" ;
      }
      out << "\n\tWeights: ";
      if(layer.m_out_layer != nullptr) {
        for(int i=0; i<layer.m_size*layer.m_out_size; i++) {
            out << layer.weights[i] << "\t" ;
        }
      } else {
          out << "\tNo weights defined" << endl;
      }
      return out;
    }
};



int main(int argc, char **argv)
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

    //NeuronLayer<cl_float> out_layer(10000, nullptr);
    //NeuronLayer<cl_float> hidden_layer(40000, &out_layer);
    //NeuronLayer<cl_float> input_layer(2000, &hidden_layer);

    NeuronLayer<cl_float> out_layer(1, nullptr);
    NeuronLayer<cl_float> hidden_layer(3, &out_layer);
    NeuronLayer<cl_float> input_layer(4, &hidden_layer);

    out_layer.setValues({1.});
    hidden_layer.setValues({0., 0., 0.});
    input_layer.setValues({ 1., 2., 3., 4.});
    input_layer.setWeights({1., 2., 3.,
                           4., 5., 6.,
                           7., 8., 9.,
                           10., 11., 12.});
    hidden_layer.setWeights({1., 2., 3.});
    cout << "In Layer: \n" << input_layer << endl;
    cout << "\nHidden Layer:\n " << hidden_layer << endl;
    cout << "\nOut Layer: \n" << out_layer << endl;

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
    //cout << "\nHidden Layer(computed): " << endl;
    //hidden_layer.display();

    out_layer.enqueueReadBuffers(queue);
    cout << "\nOut Layer(computed): \n" << out_layer << endl;

    end = std::chrono::system_clock::now();


    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "\n\nfinished computation at " << std::ctime(&end_time)
        << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}
