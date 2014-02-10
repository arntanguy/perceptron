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
        Perceptron(cl::Context& context, cl::CommandQueue& queue) : mContext(context), mQueue(queue), mFirstLayer(nullptr), mCurrentLayer(nullptr){
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

            NLayer *neuronLayer = new NLayer(size);
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
                layer->enqueueWriteBuffers(mQueue);
                layer = layer->getNextLayer();
            }
        }

        void run(cl::Kernel& kernel) {
            if(mFirstLayer == nullptr) throw std::runtime_error("No layers!");

            NLayer* layer = mFirstLayer;
            while(layer->getNextLayer() != nullptr) {
                layer->enqueueRun(kernel, mQueue);
                layer->getNextLayer()->enqueueReadBuffers(mQueue);
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

        void train(const std::vector<std::list<T>>& training_in_values, const std::vector<std::list<T>>& training_out_values) {
            if(training_in_values.size() != training_out_values.size()) {
                throw std::runtime_error("Perceptron::Train - Training input and output size must match!");
            }
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 eng(rd()); // seed the generator
            std::uniform_int_distribution<int> distr(0, training_in_values.size());

            bool train = true;
            while(train) {
                // Pick random values in training set
                int rand_training_set = distr(eng);
                const std::list<T>& training_in = training_in_values[rand_training_set];
                const std::list<T>& training_out = training_out_values[rand_training_set];
                cout << "Training with input " << training_in << endl;
                cout << "Training with output " << training_out << endl;
            }
        }
};

template<typename T> int Perceptron<T>::layerCount = 0;

#endif
