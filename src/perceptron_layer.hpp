#ifndef __PERCEPTRON_LAYER_HPP__
#define __PERCEPTRON_LAYER_HPP__


#include <initializer_list>
#include <iostream>
#include <random>
#include <sstream>
#include "openCLUtilities.hpp"
#include "exception.hpp"
#include <list>

using std::ostream;
using std::cout;
using std::endl;

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
 *
 * Be careful about memory usage
 * - Weights are only created when two layers are fully linked (see setOutputLayer)
 * - Buffers are only created if weights are defined
 * - Buffers are only uploaded to the GPU by calling enqueueWriteBuffers
 */
template<typename T>
class NeuronLayer
{
    typedef NeuronLayer<T> NLayer;

    private:
        cl::CommandQueue command_queue;
        cl::Buffer buf_values;
        cl::Buffer buf_weights;

        int mLayerNumber = 0;

        const cl_int m_size;
        cl_int m_out_size = 0;


        // Linked list
        // Next layer
        NLayer* m_out_layer;
        // Previous layer
        NLayer* m_in_layer;

        void init(NeuronLayer *in_layer, NeuronLayer *out_layer) 
        {
            m_in_layer = in_layer;
            m_out_layer = out_layer;
            // Init values to 0
            values = new T[m_size]();
            // Bias
            values[m_size-1] = 1;
            setOutputLayer(out_layer);
        }
    public:
        T* values = nullptr;
        // Weights to the next layer
        T* weights = nullptr;

        NeuronLayer(const cl_int& in_s, const cl::CommandQueue& queue, NeuronLayer* in_layer, NeuronLayer *out_layer) : command_queue(queue), m_size(in_s+1) {
            init(in_layer, out_layer);
        }

        NeuronLayer(const cl_int& in_s, const cl::CommandQueue& queue) : command_queue(queue), m_size(in_s+1), m_out_size(0) {
            init(nullptr, nullptr);
        }

        void setNumber(int id) {
            mLayerNumber = id;
        }
        void setInputLayer(NeuronLayer *in_layer) {
            m_in_layer = in_layer;
        }

        void setOutputLayer(NeuronLayer *out_layer) {
            m_out_layer = out_layer;
            if(out_layer != nullptr) {
                const cl_int& out_size = out_layer->getSize();
                weights = new T[m_size*out_size];
                m_out_size = out_size;
            } else {
                m_out_size = 0;
            }
        }


        void initRandomWeights(const float& min = -0.5, const float& max = 0.5) {
            if(m_out_size == 0) throw LayerNotLinkedException();

            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_real_distribution<float> distr(min, max);

            for(int i=0; i< m_size * m_out_size; i++) {
                weights[i] = distr(eng);
            }
        }

        NLayer* getNextLayer() {
            return m_out_layer;
        }
        
        NLayer* getPreviousLayer() {
            return m_in_layer;
        }

        cl_int getSize() const {
            return m_size;
        }

        cl_int getNbWeights() const  {
            return m_size * m_out_size;
        }

        void setValues(const std::vector<T>& init) {
            cout << "setValues" << endl;
            // Do not replace bias
            if(init.size() != m_size-1) {
                throw std::runtime_error("Your initializer list for values exceeds the maximum size!");
            }

            int j=0;
            for(const auto& i: init) {
                values[j++] = i;
            }
            // bias
            values[m_size-1] = 1;

            cout << "FUUUUUUUUUUUUUUUUUCK" << endl;
            for (int i=0; i<m_size; i++)
            {
                cout << values[i] << endl;
            }
        }

        void setValues(const std::list<T>& init) {
            cout << "setValues" << endl;
            // Do not replace bias
            if(init.size() != m_size-1) {
                throw std::runtime_error("Your initializer list for values exceeds the maximum size!");
            }

            int j=0;
            for(const auto& i: init) {
                values[j++] = i;
            }
            // bias
            values[m_size-1] = 1;

            //cout << "Setting value" << endl;
            //for (int i=0; i<m_size; i++)
            //{
            //    cout << values[i] << endl;
            //}
        }

        void uploadInputValues() {
            command_queue.enqueueWriteBuffer(buf_values, CL_TRUE, 0, sizeof(T)*m_size, values);
        }

        void setWeights(const std::list<T>& weights_list) {
            cout << "MSIZE: " << m_size << endl;
            cout << "OSIZE: " << m_out_size << endl;
            if(m_out_layer != nullptr && weights_list.size() != m_size * (m_out_size-1)) {
                throw std::runtime_error("Your initializer list for weights exceeds the maximum size!");
            }

            int j=0;
            for(auto it = begin(weights_list); it != end(weights_list); it++)
            {
                weights[j++] = *it;
            }
        }

        /**
         * @brief Prepares the buffer (host size)
         * To be done after the links between layers are set up
         */
        void createBuffers(cl::Context& context)
        {
            // Creates buffer on the device
            buf_values = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * m_size);
            buf_weights = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(T) * m_size * m_out_size);
        }

        void enqueueWriteBuffers()
        {
            std::cout << "enqueueWriteBuffers, m_size: "<< m_size<<" m_out_size: "<< m_out_size<<
                ", weight size: "<< m_size*m_out_size<<endl;
            // Prepare device memory for each layer
            command_queue.enqueueWriteBuffer(buf_values, CL_TRUE, 0, sizeof(T)*m_size, values);
            command_queue.enqueueWriteBuffer(buf_weights, CL_TRUE, 0, sizeof(T)*m_out_size*m_size, weights);
        }

        void enqueueWriteInputBuffer(const std::vector<T>& input_values)
        {
            command_queue.enqueueWriteBuffer(buf_values, CL_TRUE, 0, sizeof(T)*m_size, input_values.data());
        }

        void enqueueReadBuffers()
        {
            command_queue.enqueueReadBuffer(buf_values, CL_TRUE, 0, sizeof(T)*m_size, values);
            command_queue.enqueueReadBuffer(buf_weights, CL_TRUE, 0, sizeof(T)*m_size*m_out_size, weights);
        }

        // Should only be called by run (existence of last element not checked)
        cl::Buffer getValuesBuf() const {
            return buf_values;
        }
        cl::Buffer getWeightsBuf() const {
            return buf_weights;
        }

        void enqueueRun(cl::Kernel &kernel) {
            if(m_out_layer != nullptr) {
                kernel.setArg(0, m_size);
                kernel.setArg(1, m_out_size-1);
                kernel.setArg(2, buf_values);
                kernel.setArg(3, buf_weights);
                kernel.setArg(4, m_out_layer->getValuesBuf());
                cout << "Setting up kernel with ND-range " << m_out_size << endl;
                command_queue.enqueueNDRangeKernel(kernel, cl::NullRange,cl::NDRange(m_out_size-1),cl::NullRange);
                command_queue.finish();
            } else {
                throw std::runtime_error("Can't run kernel on a null layer!");
            }
        }

        void enqueueTrainOutputLayer(cl::Kernel &kernel, cl::Buffer& expected_out_buf, cl::Buffer& delta_out_buf) {

            cout << "enqueueTrainOutputLayer: size " << m_size << endl;
            kernel.setArg(0, buf_values);
            kernel.setArg(1, expected_out_buf);
            kernel.setArg(2, delta_out_buf);
            command_queue.enqueueNDRangeKernel(kernel, cl::NullRange,cl::NDRange(m_size-1),cl::NullRange);
            if(command_queue.finish()!= CL_SUCCESS) {
                throw std::runtime_error("PerceptronLayer::enqueueTrainOutputLayer - command queue failed to execut");
            }
        }

        void enqueueTrainBackpropagate(cl::Kernel &kernel, cl::Buffer& delta_out_buf, cl::Buffer& succ_delta_buf) {
            if(m_out_layer != nullptr) {
                cout << "enqueueTrainBackpropagate:: m_size=" << m_size << ", m_out_layer_size= " << m_out_layer->getSize() << endl;
                kernel.setArg(0, m_size);
                kernel.setArg(1, m_out_layer->getSize());
                kernel.setArg(2, buf_values);
                kernel.setArg(3, buf_weights);
                kernel.setArg(4, succ_delta_buf);
                kernel.setArg(5, delta_out_buf);
                cout  << "PerceptronLayer::enqueueTrainBackpropagate - running kernel" << endl;
                command_queue.enqueueNDRangeKernel(kernel, cl::NullRange,cl::NDRange(m_size-1),cl::NullRange);
                command_queue.finish();
            } else {
                throw std::runtime_error("Can't run kernel on a null layer!");
            }
        }

        void enqueueTrainUpdateWeights(cl::Kernel& kernel, cl::Buffer& delta_buf)
        {
            NLayer* prev_layer = getPreviousLayer();
            if(prev_layer != nullptr) {
                kernel.setArg(0, prev_layer->getSize());
                kernel.setArg(1, prev_layer->getValuesBuf());
                kernel.setArg(2, delta_buf);
                kernel.setArg(3, prev_layer->getWeightsBuf());
                cout  << "PerceptronLayer::enqueueTrainWeights - running kernel with NDRange " << m_size*(m_out_size-1) << endl;
                if(command_queue.enqueueNDRangeKernel(kernel, cl::NullRange,cl::NDRange((m_size-1)*(prev_layer->getSize())),cl::NullRange) != CL_SUCCESS) throw "fuck";
                command_queue.finish();
            } else {
                throw std::runtime_error("Can't run kernel on a null layer!");
            }
        }

        friend ostream& operator<< (ostream &out, const NeuronLayer& layer) {
            out << "Displaying Layer " << layer.mLayerNumber << endl;
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

#endif
