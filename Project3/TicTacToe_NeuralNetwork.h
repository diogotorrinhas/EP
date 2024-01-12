#ifndef NN_BASE_H
#define NN_BASE_H

typedef struct
{
    int number_I; //Number of Input Units
    int number_H; //Number of Hidden Units
    int number_O; //Number of Output Units
    double** weights_to_hidden; //I->H
    double** weights_to_output; //H->O
    double* I;  // Array to store input values
    double* H; // Array to store hidden layer values
    double* O; // Array to store output value
    double* bias_to_hidden; // Array to store the weights of the bias to the hidden layer
    double* bias_to_output; // Array to store the weights of the bias to the output layer
    double* errors_output; // Array to store the errors of the output layer
    double* inp_hidden; // Array to store the input values of each hidden unit in the hidden layer
    double* inp_output; // Array to store the input values of each output unit in the output layer
    double* delta_hidden; // // Array to store the delta values of each hidden unit in the hidden layer
    double* delta_output; // Array to store the delta values of each output unit in the output layer
} NeuralNetwork;

NeuralNetwork* nn_create();
void load_initial_bias_weights_to_outputLayer(double values[], NeuralNetwork *nn);
void load_initial_bias_weights_to_hiddenLayer(double values[], NeuralNetwork *nn);
void propagate(NeuralNetwork *nn);
void back_propagate(NeuralNetwork *nn, double learning_rate);
void load_input_values(double input_values[], NeuralNetwork *nn);
void load_weights(NeuralNetwork *nn);
void load_nn(NeuralNetwork *nn);
void write_nn(NeuralNetwork *nn);
void nn_destroy(NeuralNetwork *nn);


#endif