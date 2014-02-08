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
void kernel perceptron(global const int* in_layer_size, global const int* out_layer_size, global const float *in_value, global const float* in_weights, global float* out_values)
{
    private const int global_id = get_global_id(0);
    private const int out_layer_s = *out_layer_size;
    private const int in_layer_s = *in_layer_size;
    private const int offset = out_layer_s * global_id;

    private float sum = 0.;
    for(int i=0; i < in_layer_s; i++) {
        sum += in_weights[i*out_layer_s+global_id] * in_value[i];
    }
    //out_values[global_id] = sigma(sum);
    out_values[global_id] = sum;
}
