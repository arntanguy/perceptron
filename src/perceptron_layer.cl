void kernel perceptron(global const int* layer_size, global const int* previous_layer_size, global const float *in_value, global const float* in_weights, global float* out_values)
{

    private const int global_id = get_global_id(0);
    private const int layer_s = *layer_size;
    private const int previous_s = *previous_layer_size;
    private const int offset = layer_s * global_id;

    private float sum = 0.;
    for(int i=0; i < previous_s; i++) {
        sum += in_weights[i*layer_s+global_id] * in_value[i];
    }

    out_values[global_id] = sum;
}
