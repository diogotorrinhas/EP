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
} NeuralNetwork;

NeuralNetwork* nn_create();
void propagate(double input_values[], NeuralNetwork *nn);
void load_input_values(double input_values[], NeuralNetwork *nn);
void load_nn(NeuralNetwork *nn);
void write_nn(NeuralNetwork *nn);
void nn_destroy(NeuralNetwork *nn);


#endif