#ifndef __PERCEPTRON_HPP__
#define __PERCEPTRON_HPP__

#include "perceptron_layer.hpp"
#include "debug/prettyprint.hpp"

#include <list>
#include <vector>
#include <random>

template<typename T>
class Perceptron
{
    private:
        cl::Context mContext;
        cl::CommandQueue mQueue;

        typedef NeuronLayer<T> NLayer;
        NLayer *mFirstLayer;
        NLayer *mCurrentLayer;

        static int layerCount;
        int mCurrentLayerNumber = 0;

    public:
        Perceptron(cl::Context& context, cl::CommandQueue& queue) : mContext(context), mQueue(queue), mFirstLayer(nullptr), mCurrentLayer(nullptr) {
            mCurrentLayerNumber = layerCount++;
        }

        void initRandomWeights() {
            NLayer *layer = mFirstLayer;
            while(layer->getNextLayer() != nullptr) {
                layer->initRandomWeights();
            }
        }

        void createLayer(const int& size) {
            cout << "createLayer: " << mCurrentLayerNumber << " of size " << size << endl;

            NLayer *neuronLayer = new NLayer(size, mQueue);
            cout << "createLayer: " << neuronLayer << endl;
            if(mFirstLayer == nullptr) {
                mFirstLayer = neuronLayer;
                mCurrentLayer = mFirstLayer;
            } else {
                cout << "Level "<<mCurrentLayerNumber << ", neuronLayer: " << neuronLayer << endl;
                cout << mCurrentLayer << " -> " << neuronLayer << endl;
                mCurrentLayer->setOutputLayer(neuronLayer);
                mCurrentLayer->initRandomWeights();
                mCurrentLayer->createBuffers(mContext);
                // Move to the next layer
                neuronLayer->setInputLayer(mCurrentLayer);
                mCurrentLayer = neuronLayer;
            }
            mCurrentLayerNumber++;
        }

        void setWeights(const std::list<std::list<T>>& weights)
        {
            NLayer *layer = mFirstLayer;
            auto it = begin(weights);
            for(; it != end(weights); it++) {
                if(layer == nullptr) throw "Null layer, you're trying to set to many weights!";
                
                layer->setWeights(*it);
                layer = layer->getNextLayer();
            }
        }

        void setInputValues(const std::list<T>& values) 
        {
            if(mFirstLayer == nullptr) throw "Perceptron::setInputValues - null layer";

            mFirstLayer->setValues(values);
        }

        void upload() {
            cout << "upload " << endl;

            // Create the buffers for the last layer
            mCurrentLayer->createBuffers(mContext);

            NLayer *layer = mFirstLayer;
            while(layer != nullptr) {
                layer->enqueueWriteBuffers();
                layer = layer->getNextLayer();
            }
        }

        void run(cl::Kernel& kernel) {
            if(mFirstLayer == nullptr) throw std::runtime_error("No layers!");

            NLayer* layer = mFirstLayer;
            while(layer->getNextLayer() != nullptr) {
                layer->enqueueRun(kernel);
                layer->getNextLayer()->enqueueReadBuffers();
                //cout << "\nLayer " << layer->getNextLayer() << " (computed): \n" << *layer->getNextLayer() << endl;
                layer = layer->getNextLayer();
            }
        }

        void displayAll() {
            NLayer* layer = mFirstLayer; 
            while(layer != nullptr) {
                cout << layer << " -> ";
                layer = layer->getNextLayer();
            }
            cout << endl;
            layer = mFirstLayer; 
            while(layer != nullptr) {
                cout << *layer << endl;
                layer = layer->getNextLayer();
            }
            cout << endl;
        }
        NeuronLayer<T>* getFirstLayer() {
            return mFirstLayer;
        }

        NeuronLayer<T>* getLastLayer() {
            return mCurrentLayer;
        }

        void train(cl::Kernel& kernel, cl::Kernel& train_output_layer_kernel, const std::vector<std::vector<T>>& training_in_values, const std::vector<std::vector<T>>& training_out_values) {
            // XXX: nothing to ensure weights have been initialized to [-0.5, 0.5]
            if(training_in_values.size() != training_out_values.size()) {
                throw std::runtime_error("Perceptron::Train - Training input and output size must match!");
            } else if(mFirstLayer == nullptr) {
                throw std::runtime_error("Perceptron::Train - You must have more than one layer to train a perceptron !");
            }
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<int> distr(0, training_in_values.size()-1);

            int train = 0;
            while(train++ < 2) {
                // Pick random values in training set
                int rand_training_set = distr(eng);
                const std::vector<T>& training_in = training_in_values[rand_training_set];
                const std::vector<T>& training_out = training_out_values[rand_training_set];
                cout << "Training with input " << rand_training_set << ": " << training_in << endl;
                cout << "Training with output " << rand_training_set << ": " << training_out << endl;

                /**
                 * Step 1.0: Prepare buffers
                 **/
                // XXX: creates buffer each time (shouldn't it be created first and used when needed
                // To see according to how much memory available
                // Solution simple: vector of buffers corresponding to the training set size
                cl::Buffer training_out_buf = cl::Buffer(mContext, CL_MEM_READ_ONLY, sizeof(T) * training_out.size());
                cl::Buffer delta_out_buf = cl::Buffer(mContext, CL_MEM_READ_ONLY, sizeof(T) * training_in.size());

                /**
                 * Step 1.1: Compute output o
                 **/
                cout << "Writing input to GPU" << endl;
                // Write input data to GPU. Leave all other parameters unchanged
                mFirstLayer->enqueueWriteInputBuffer(training_in);

                // Running perceptron kernel to compute the output layer o
                run(kernel);
                
                /**
                 * Step 1.2: Compute delta_i for the output layer
                 **/
                cout << "Computing delta_i for last layer" << endl;
                mQueue.enqueueWriteBuffer(training_out_buf, CL_TRUE, 0, sizeof(T)*training_out.size(), training_out.data());
                // expected out, delta
                mCurrentLayer->enqueueTrainOutputLayer(train_output_layer_kernel, training_out_buf, delta_out_buf);
                T *delta_values = new T[training_out.size()];
                delta_values[0] = 0;
                mQueue.enqueueReadBuffer(delta_out_buf, CL_TRUE, 0, sizeof(T)*training_out.size(), delta_values);
                
                cout << "delta_i = " ;
                for(int i=0; i<training_out.size(); i++) {
                    cout << delta_values[i] << ", " << endl;
                }

                cout << endl;
            }
        }
};

template<typename T> int Perceptron<T>::layerCount = 0;

#endif
