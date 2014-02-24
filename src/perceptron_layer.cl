/******************************************************************************
*     Copyright (C) 2014 TANGUY Arnaud arn.tanguy@gmail.com                   *
*                                                                             *
* This program is free software; you can redistribute it and/or modify        *
* it under the terms of the GNU General Public License as published by        *
* the Free Software Foundation; either version 2 of the License, or           *
* (at your option) any later version.                                         *
*                                                                             *
* This program is distributed in the hope that it will be useful,             *
* but WITHOUT ANY WARRANTY; without even the implied warranty of              *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                *
* GNU General Public License for more details.                                *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc.,     *
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.                 *
 ******************************************************************************/

float sigma(float x)
{
    return 1./(1. + exp(-x));
}

/**
 * @brief Computes delta for all of the output neurons.
 * 
 * @param values
 *      Values of the output layer
 * @param expected_values
 *      Values expected as output of the perceptron
 * @param delta
 *      Output of the function: computes the delta needed for the training algorithm
 **/
void kernel perceptron_train_output_layer(
        global const float* values,
        global const float* expected_values,
        global float* delta)
{
    private const float ci = expected_values[get_global_id(0)];
    private const float oi = values[get_global_id(0)];
    // Equivalent to sigma'(yi) * (ci-oi)
    delta[get_global_id(0)] = oi * (1-oi) * (ci-oi); 
}

/**
 * @brief Computes delta for all layers (but the last one) 
 * 
 * @param succ_layer_size
 *      Size of the output layer of current layer 
 * @param current_layer_values 
 *      Values expected as output of the perceptron
 * @param delta
 *      Output of the function: computes the delta needed for the training algorithm
 **/
void kernel perceptron_train_backpropagate(
        const int succ_layer_size,
        global const float* current_layer_values,
        global const float* weights,
        global const float* succ_layer_delta_i,
        // output
        global float* current_delta_out
        )
{
    private const int i = get_global_id(0);
    private const float oi = current_layer_values[get_global_id(0)];
    private const int succ_size = succ_layer_size;
    private const int weight_offset = i * succ_size;

    private float sum = 0.f;
    for(int k=0; k < succ_size; k++) {
        sum += succ_layer_delta_i[k] * weights[weight_offset + k];
    }
    printf("i: %i\n", i);
    printf("succ_size %i\n", succ_size);
    printf("oi %f\n", oi);
    printf("sum %f\n", sum);
    printf("delta_i %f\n", oi * (1-oi) * sum);
    printf("\n");
    current_delta_out[i] = oi*(1-oi) * sum;
}

void kernel perceptron_train_update_weights(
        const int out_layer_size,
        global const float *neuron_values,
        global const float *delta,
        global float* weights)
{
    private const int global_id = get_global_id(0);
    private const int out_layer_s = out_layer_size;
    private const float val = neuron_values[global_id % out_layer_s];

    // XXX to change
    private const float epsilon = 1.;
    private float sum = 0.;
    // For each weight
    weights[global_id] = weights[global_id] + epsilon * delta[global_id] * val; 
}

/**
* @brief Computes one layer of the perceptron given the previous one and the
* weights
* The kernel is run once for each layer.
* The work items are each tasked with computing the output of a single neuron
* of the out layer.
*
* @param out_layer_size
*   Size of the output layer (number of elements in the output array that will
*   contain the result for each neuron).
* @param in_layer_size
*   Number of elements of the input layer
* @param in_value
*   Values of the neuron in the previous layer
* @param in_weights
*   Array containing the weights for each input neuron. It is organised as a
*   two dimensional matrix, written by concatenating each line in the array
*   [ w11, w12, w13, ...
*     w21, w22, w23, ...
*     ..., ..., ..., ...
*   ]
*   Where wij is the weight linking the neuron i of the input layer to the
*   neuron j of the output layer
* @param out_values
*   Computed values for the current layer
*/
void kernel perceptron(
        const int in_layer_size,
        const int out_layer_size,
        global const float *in_value,
        global const float* in_weights,
        global float* out_values)
{
    private const int global_id = get_global_id(0);
    private const int out_layer_s = out_layer_size;
    private const int in_layer_s = in_layer_size;

    private float sum = 0.;
    for(int i=0; i < in_layer_s; i++) {
        sum += in_weights[i*out_layer_s+global_id] * in_value[i];
    }
    out_values[global_id] = sigma(sum);
    //out_values[global_id] = sum;
}
