#ifndef __PERCEPTRON_HPP__
#define __PERCEPTRON_HPP__

#include "perceptron_layer.hpp"
#include "debug/prettyprint.hpp"

#include <list>
#include <vector>
#include <random>
#include <stack>

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
                layer = layer->getNextLayer();
            }
        }

        void createLayer(const int& size) {
            NLayer *neuronLayer = new NLayer(size, mQueue);
            if(mFirstLayer == nullptr) {
                mFirstLayer = neuronLayer;
                mCurrentLayer = mFirstLayer;
            } else {
                mCurrentLayer->setOutputLayer(neuronLayer);
                mCurrentLayer->initRandomWeights();
                mCurrentLayer->createBuffers(mContext);
                // Move to the next layer
                neuronLayer->setInputLayer(mCurrentLayer);
                mCurrentLayer = neuronLayer;
            }
            neuronLayer->setNumber(mCurrentLayerNumber);
            mCurrentLayerNumber++;
        }

        void setWeights(const std::list<std::list<T>>& weights)
        {
            NLayer *layer = mFirstLayer;
            auto it = begin(weights);
            for(; it != end(weights); it++) {
                if(layer == nullptr) throw "Null layer, you're trying to set to many weights!";
                
                layer->setWeights(*it);
                layer->enqueueWriteBuffers();
                layer = layer->getNextLayer();
            }
        }

        void setInputValues(const std::list<T>& values) 
        {
            if(mFirstLayer == nullptr) throw "Perceptron::setInputValues - null layer";

            mFirstLayer->setValues(values);
            mFirstLayer->uploadInputValues();
        }

        void upload() {
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
                layer = layer->getNextLayer();
            }
        }

        void enqueueReadAllBuffers()
        {
            if(mFirstLayer == nullptr) return; 

            NLayer* layer = mFirstLayer;
            while(layer != nullptr) {
                layer->enqueueReadBuffers();
                layer = layer->getNextLayer();
            }
        }

        void displayAll() {
            NLayer* layer = mFirstLayer; 
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

        float maxError(const std::vector<T>& expected, float confidence)
        {
            mCurrentLayer->enqueueReadValues();
            T *values = mCurrentLayer->getValues();
            float max_error = 0.;
            for(int i=0; i<mCurrentLayer->getSize()-1; i++) {
                max_error = std::fmax(max_error, std::fabs(values[i] - expected[i]));
            }
            return max_error;
        }
        bool checkOutputAgainstConfidence(const std::vector<T>& expected, float confidence)
        {
            mCurrentLayer->enqueueReadValues();
            T *values = mCurrentLayer->getValues();
            float max_error = 0.;
            for(int i=0; i<mCurrentLayer->getSize()-1; i++) {
                max_error = std::fmax(max_error, std::fabs(values[i] - expected[i]));
            }
            cout << "max error: " << max_error << " <= " << 1.f-confidence << endl;
            return max_error <= (1.f-confidence);

        }

        void train(cl::Kernel& kernel, cl::Kernel& train_output_layer_kernel, cl::Kernel& train_backpropagate_kernel, cl::Kernel& train_update_weights_kernel, const std::vector<std::vector<T>>& training_in_values, const std::vector<std::vector<T>>& training_out_values, const float& epsilon, const float& confidence=0.8, const int& max_iterations=100000) {
            // XXX: nothing to ensure weights have been initialized to [-0.5, 0.5]
            if(training_in_values.size() != training_out_values.size()) {
                throw std::runtime_error("Perceptron::Train - Training input and output size must match!");
            } else if(mFirstLayer == nullptr) {
                throw std::runtime_error("Perceptron::Train - You must have more than one layer to train a perceptron !");
            }
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<int> distr(0, training_in_values.size()-1);

            /**
             * Prepare buffers
             **/
            std::vector<cl::Buffer> delta_bufs;
            NLayer *layer = mFirstLayer;
            while(layer != nullptr) {
                delta_bufs.push_back(cl::Buffer(mContext, CL_MEM_READ_ONLY, sizeof(T) * layer->getSize()));
                layer = layer->getNextLayer();
            }
            cl::Buffer& delta_out_buf = *(--end(delta_bufs)); 
            cl::Buffer training_out_buf(mContext, CL_MEM_READ_ONLY, sizeof(T) * training_out_values.size());



            int train = 0;
            bool hasConverged=false;
            while(train++ < max_iterations && !hasConverged) {
                // Pick random values in training set
                int rand_training_set = distr(eng);
                //const std::vector<T>& training_in = training_in_values[rand_training_set];
                //const std::vector<T>& training_out = training_out_values[rand_training_set];
                const std::vector<T>& training_in = training_in_values[(train-1)%4];
                const std::vector<T>& training_out = training_out_values[(train-1)%4];

                /**
                 * Step 1.1: Compute output o
                 **/
                // Write input data to GPU. Leave all other parameters unchanged
                mFirstLayer->setValues(training_in);
                mFirstLayer->uploadInputValues();

                // Running perceptron kernel to compute the output layer o
                run(kernel);


                /**
                 * Checks every 100 iteration if 
                 * algorithm has converged with confidence greater than minimum required
                 **/
                if(train%100 == 0) {
                    hasConverged = true;
                    for(int i=0; i<training_in_values.size(); i++) {
                        const auto& in_values = training_in_values[i];
                        const auto& out_values = training_out_values[i];
                        mFirstLayer->setValues(in_values);
                        mFirstLayer->uploadInputValues();

                        run(kernel);

                        float max_error = maxError(out_values, confidence);
                        cout << "max_error: " << max_error << " for training value " << i  << " at iteration " << train << endl;
                        if(max_error > 1.f-confidence) {
                            hasConverged = false;
                            mFirstLayer->setValues(training_in);
                            mFirstLayer->uploadInputValues();
                            break;
                        } 
                    }
                }
                if(hasConverged) {
                    cout << "Trained in " << train << " iterations." << endl;
                    return;
                }

                /**
                 * Step 1.2: Compute delta_i for the output layer
                 **/
                // Upload expected output to GPU
                mQueue.enqueueWriteBuffer(training_out_buf, CL_TRUE, 0, sizeof(T)*training_out.size(), training_out.data());
                // expected out, delta
                mCurrentLayer->enqueueTrainOutputLayer(train_output_layer_kernel, training_out_buf, delta_out_buf);
                /**
                 * ----------------
                 * Back propagation
                 * ----------------
                 **/
                // second to last buffer first
                int current_buf_num = delta_bufs.size()-1;
                // Start from second to last layer and move up the layers to the first one
                NLayer* layer = mCurrentLayer->getPreviousLayer();
                while(layer != nullptr) {
                    cl::Buffer& succDeltaBuffer = delta_bufs[current_buf_num];
                    cl::Buffer& currentDeltaBuffer = delta_bufs[--current_buf_num]; 
                    layer->enqueueTrainBackpropagate(train_backpropagate_kernel, currentDeltaBuffer, succDeltaBuffer);
                    layer = layer->getPreviousLayer();
                }

                /**
                 * Update the weights
                 **/
                current_buf_num = 0; 
                layer = mFirstLayer->getNextLayer();
                while(layer != nullptr) {
                    cl::Buffer & buf = delta_bufs[++current_buf_num];
                    layer->enqueueTrainUpdateWeights(train_update_weights_kernel, buf, epsilon);
                    layer = layer->getNextLayer();
                }
                
            }
        }
};

template<typename T> int Perceptron<T>::layerCount = 0;

#endif
