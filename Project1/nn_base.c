#include <stdio.h>
#include <stdlib.h>
#include "nn_base.h"

NeuralNetwork* nn_create(){
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    FILE *fp = fopen("NeuralNetwork1.txt", "r");
    int number_inputs, number_hidden, number_outputs;

    // Read the first line (I H O)
    if (fscanf(fp, "%d %d %d", &number_inputs, &number_hidden, &number_outputs) == 3) {
        nn->number_I = number_inputs;
        nn->number_H = number_hidden;
        nn->number_O = number_outputs;
        fclose(fp);
    }

    //Allocate memory for input, hidden, and output layers
    nn->I = malloc(sizeof(double) * nn->number_I);
    nn->H = malloc(sizeof(double) * nn->number_H);
    nn->O = malloc(sizeof(double) * nn->number_O);

    //Initialize weights
    nn->weights_to_hidden = (double**)malloc(nn->number_I * sizeof(double*));
    nn->weights_to_output = (double**)malloc(nn->number_H * sizeof(double*));

    //Initialize weight matrices and layers to zero
    for (int i = 0; i < nn->number_I; i++) {
        nn->weights_to_hidden[i] = (double*)malloc(nn->number_H * sizeof(double));
        for (int j = 0; j < nn->number_H; j++) {
            nn->weights_to_hidden[i][j] = 0.0;
        }
    }

    for (int i = 0; i < nn->number_H; i++) {
        nn->weights_to_output[i] = (double*)malloc(nn->number_O * sizeof(double));
        for (int j = 0; j < nn->number_O; j++) {
            nn->weights_to_output[i][j] = 0.0;
        }
    }

    return nn;
}


void propagate(double input_values[], NeuralNetwork *nn){
    //Propagate input to hidden layer
    for(int i = 0; i < nn->number_H; i++){
        double aux = 0.0;
        for(int j = 0; j< nn->number_I; j++){
            aux += nn->I[j] * nn->weights_to_hidden[j][i];
        }
        nn->H[i] = aux;
    }

    //Propagate hidden to output layer
    for(int i = 0; i<nn->number_O; i++){
        double aux = 0.0;
        for(int j = 0; j<nn->number_H; j++){
            aux += nn->H[j] * nn->weights_to_output[j][i];
        }
        nn->O[i] = aux;
    }
}

void load_input_values(double input_values[], NeuralNetwork *nn){
    //Give the input values to the input layer
    for(int i = 0; i < nn->number_I; i++){
        nn->I[i] = input_values[i];
    }
}

void load_nn(NeuralNetwork *nn){
    FILE *fp = fopen("NeuralNetwork1.txt", "r");
    int fl,sl, input_unit, hidden_unit;
    double weight;

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // Skip the first line because it has already been readed in the method nn_create(I H O)
    char line[256];
    fgets(line, sizeof(line), fp);

    // Read the remaining lines with weight values
    while (fscanf(fp, "%d:%d %d:%d %lf", &fl, &input_unit, &sl, &hidden_unit, &weight) == 5) {
        printf("First input: %d, Second input: %d\n", input_unit, hidden_unit);
        if (fl == 1 && sl == 2) {
            nn->weights_to_hidden[input_unit - 1][hidden_unit - 1] = weight;
        } else if(fl == 2 && sl == 3){
            nn->weights_to_output[input_unit - 1][hidden_unit - 1] = weight;
        }
    }

    fclose(fp);
}

void write_nn(NeuralNetwork *nn){
    FILE *fp = fopen("NeuralNetworkOutput.txt", "w");

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(fp, "%d %d %d\n", nn->number_I, nn->number_H, nn->number_O);

    for (int i = 0; i < nn->number_I; i++) {
        for (int j = 0; j < nn->number_H; j++) {
            fprintf(fp, "1:%d 2:%d %lf\n", i + 1, j + 1, nn->weights_to_hidden[i][j]);
        }
    }

    for (int i = 0; i < nn->number_H; i++) {
        for (int j = 0; j < nn->number_O; j++) {
            fprintf(fp, "2:%d 3:%d %lf\n", i + 1, j + 1, nn->weights_to_output[i][j]);
        }
    }

    //Output values of the output layer
    // for(int i = 0; i<nn->number_O; i++){
    //     fprintf(fp, "3:%d %lf\n", i+1, nn->O[i]);
    // }

    // fclose(fp);
}

void nn_destroy(NeuralNetwork *nn){
    if(nn){
        for (int i = 0; i < nn->number_I; i++) {
            free(nn->weights_to_hidden[i]);
        }
        free(nn->weights_to_hidden);

        for (int i = 0; i < nn->number_H; i++) {
            free(nn->weights_to_output[i]);
        }
        free(nn->weights_to_output);

        free(nn->I);
        free(nn->H);
        free(nn->O);
        free(nn);
    }
}

int main(){
    NeuralNetwork *nn = nn_create();
    load_nn(nn);
    
    //Next two prints are just used for debugging
    
    // Print the loaded weights from Input to Hidden Layer connections
    printf("Weights from Input to Hidden Layer:\n");
    for (int i = 0; i < nn->number_I; i++) {
        for (int j = 0; j < nn->number_H; j++) {
            printf("Weight I%d to H%d: %lf\n", i+1, j+1, nn->weights_to_hidden[i][j]);
        }
    }

    // Print the loaded weights from Hidden to Output Layer connections
    printf("Weights from Hidden to Output Layer:\n");
    for (int i = 0; i < nn->number_H; i++) {
        for (int j = 0; j < nn->number_O; j++) {
            printf("Weight H%d to O%d: %lf\n", i+1, j+1, nn->weights_to_output[i][j]);
        }
    }

    double input_values[] = {2.0, 1.0};
    load_input_values(input_values, nn);
    propagate(input_values, nn);
    for (int i = 0; i < nn->number_O; i++) {
        printf("Output[%d]: %lf\n",i, nn->O[i]);
    }
    write_nn(nn);

    nn_destroy(nn);
    return 0;
}
