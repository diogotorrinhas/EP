#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn_base.h"

NeuralNetwork* nn_create(int trainingMode){
    int number_inputs, number_hidden, number_outputs;
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        fprintf(stderr, "Error allocating memory for NeuralNetwork\n");
        exit(EXIT_FAILURE);
    }

    if(trainingMode == 0){ //Normal training mode, read number inputs,hidden,output from NeuralNetwork1 file
        FILE *fp = fopen("NeuralNetwork1.txt", "r");
        if (fp == NULL) {
            fprintf(stderr, "Error opening file NeuralNetwork1.txt\n");
            exit(EXIT_FAILURE); 
        }
        
        // Read the first line (I H O)
        if (fscanf(fp, "%d %d %d", &number_inputs, &number_hidden, &number_outputs) == 3) {
            nn->number_I = number_inputs + 1; //Add one bias unit to the last position of the input layer
            nn->number_H = number_hidden + 1; //Add one bias unit to the last position of the hidden layer
            nn->number_O = number_outputs;
            fclose(fp);
        }
    }else{ ///XOR training mode, read number inputs,hidden,output from XOR_NeuralNetwork file
        FILE *fp = fopen("XOR_NeuralNetwork.txt", "r");
        if (fp == NULL) {
            fprintf(stderr, "Error opening file XOR_NeuralNetwork.txt\n");
            exit(EXIT_FAILURE); 
        }

        // Read the first line (I H O)
        if (fscanf(fp, "%d %d %d", &number_inputs, &number_hidden, &number_outputs) == 3) {
            nn->number_I = number_inputs + 1; //Add one bias unit to the last position of the input layer
            nn->number_H = number_hidden + 1; //Add one bias unit to the last position of the hidden layer
            nn->number_O = number_outputs;
            fclose(fp);
        }
    }
        

    //Allocate memory for input, hidden, and output layers
    nn->I = malloc(sizeof(double) * nn->number_I);
    nn->H = malloc(sizeof(double) * nn->number_H); 
    nn->O = malloc(sizeof(double) * nn->number_O);
    nn->bias_to_output = malloc(sizeof(double) * nn->number_O); //Alocatte memory for the bias unit to the output layer
    nn->bias_to_hidden = malloc(sizeof(double) * (nn->number_H-1)); ///Alocatte memory for the bias unit to the hidden layer
    nn->errors_output = malloc(sizeof(double) * nn->number_O);
    nn->inp_hidden = malloc(sizeof(double) * (nn->number_H-1));
    nn->inp_output = malloc(sizeof(double) * nn->number_O);
    nn->delta_hidden = malloc(sizeof(double) * (nn->number_H-1));
    nn->delta_output = malloc(sizeof(double) * nn->number_O);

    //Initialize bias unit to 1 in each layer (first and hidden layer)
    nn->H[nn->number_H-1] = 1.0;
    nn->I[nn->number_I-1] = 1.0;

    //Initialize weights
    nn->weights_to_hidden = (double**)malloc((nn->number_I-1) * sizeof(double*));
    nn->weights_to_output = (double**)malloc((nn->number_H-1) * sizeof(double*));

    //Initialize weight matrices and layers to zero
    for (int i = 0; i < nn->number_I-1; i++) {
        nn->weights_to_hidden[i] = (double*)malloc((nn->number_H-1) * sizeof(double));
        for (int j = 0; j < nn->number_H-1; j++) {
            nn->weights_to_hidden[i][j] = 0.0;
        }
    }

    for (int i = 0; i < nn->number_H-1; i++) {
        nn->weights_to_output[i] = (double*)malloc(nn->number_O * sizeof(double));
        for (int j = 0; j < nn->number_O; j++) {
            nn->weights_to_output[i][j] = 0.0;
        }
    }

    //Initialize weight of the bias unit (HIDDEN TO OUTPUT LAYER) to zero
    for(int i = 0; i < nn->number_O; i++) {
        nn->bias_to_output[i] = 0.0;
    }

    //Initialize weight of the bias unit (INPUT TO HIDDEN LAYER) to zero
    for(int i = 0; i < nn->number_H-1; i++){
        nn->bias_to_hidden[i] = 0.0;
    }

    //Initialize errors to zero
    for(int i = 0; i < nn->number_O; i++){
        nn->errors_output[i] = 0.0;
    }

    //Initialize inputs of the units of the hidden/output layer to zero 
    for(int i = 0; i < nn->number_H-1; i++){
        nn->inp_hidden[i] = 0.0;
    }

    for(int i = 0; i < nn->number_O; i++){
        nn->inp_output[i] = 0.0;
    }


    //Initialize both unit deltas in each layer to zero
    for(int i = 0; i < nn->number_H-1; i++){
        nn->delta_hidden[i] = 0.0;
    }

    for(int i = 0; i < nn->number_O; i++){
        nn->delta_output[i] = 0.0;
    }

    return nn;
}


//Some initial function that just add random values intruduced by the user to the weights that will be trained (changed) later
void load_initial_bias_weights_to_outputLayer(double values[], NeuralNetwork *nn){
    for(int i = 0; i < nn->number_O; i++){
        nn->bias_to_output[i] = values[i];
    }
}

void load_initial_bias_weights_to_hiddenLayer(double values[], NeuralNetwork *nn){
    for(int i = 0; i < nn->number_H-1; i++){
        nn->bias_to_hidden[i] = values[i];
    }
}

void propagate(NeuralNetwork *nn){

    //Propagate input to hidden layer using SIGMOID function
    for(int i = 0; i < nn->number_H-1; i++){
        double somatorio = 0.0;
        for(int j = 0; j< nn->number_I-1; j++){
            somatorio += nn->weights_to_hidden[j][i]*nn->I[j];
        }
        nn->inp_hidden[i] = somatorio + (1*nn->bias_to_hidden[i]); //Também podia ser -> (nn->I[nn->number_I-1] * nn->bias_to_hidden[i]) 
        nn->H[i] = 1/(1 + exp(-nn->inp_hidden[i]));
    }

    //Propagate hidden to output layer using SIGMOID function
    for(int i = 0; i<nn->number_O; i++){
        double somatorio = 0.0;
        for(int j = 0; j<nn->number_H-1; j++){
            somatorio += nn->weights_to_output[j][i]*nn->H[j];
        }
        nn->inp_output[i] = somatorio + (1*nn->bias_to_output[i]); //Também podia ser -> (nn->H[nn->number_H-1] * nn->bias_to_output[i])
        nn->O[i] = 1/(1 + exp(-nn->inp_output[i]));
    }

}

void back_propagate(NeuralNetwork *nn, double learning_rate) {
    // Update weights from hidden to output layer
    for (int i = 0; i < nn->number_O; i++) {
        nn->delta_output[i] = (1/(1 + exp(-nn->inp_output[i]))) * (1 - (1/(1 + exp(-nn->inp_output[i])))) * nn->errors_output[i]; //nn->O[i] * (1 - nn->O[i]) * nn->errors_output[i];

        // Update weights
        for (int j = 0; j < nn->number_H - 1; j++) {
            nn->weights_to_output[j][i] = nn->weights_to_output[j][i] + (learning_rate * nn->delta_output[i] * nn->H[j]); 
        }

        // Update bias
        nn->bias_to_output[i] = nn->bias_to_output[i] + (learning_rate * nn->delta_output[i] * 1);  
    }

    // Update weights from input to hidden layer
    for (int i = 0; i < nn->number_H - 1; i++) {
        double derivative = (1/(1 + exp(-nn->inp_hidden[i]))) * (1 - (1/(1 + exp(-nn->inp_hidden[i]))));  //(1/1+(e^-x))*(1-(1/1+(e^-x))    nn->H[i] * (1 - nn->H[i]);
        double somatorio = 0.0;

        // Calculate Delta Hidden
        for (int k = 0; k < nn->number_O; k++) {
            somatorio += (nn->weights_to_output[i][k] * nn->delta_output[k]);
        }

        nn->delta_hidden[i] = derivative * somatorio;

        for(int j = 0; j < nn->number_I-1; j++){
            nn->weights_to_hidden[j][i] = nn->weights_to_hidden[j][i] + (learning_rate* nn->delta_hidden[i] * nn->I[j]); 
        }

        // Update bias
        nn->bias_to_hidden[i] = nn->bias_to_hidden[i] + (learning_rate * nn->delta_hidden[i] * 1); 
    }
}

void load_input_values(double input_values[], NeuralNetwork *nn){
    //Give the input values to the input layer
    for(int i = 0; i < nn->number_I-1; i++){
        nn->I[i] = input_values[i];
    }
}

void load_weights(NeuralNetwork *nn){
    // Seed the random number generator with the current time
    srand(time(NULL));

    // Assign random weights to the edges
    for (int i = 0; i < nn->number_I - 1; i++) {
        for (int j = 0; j < nn->number_H - 1; j++) {
            //nn->weights_to_hidden[i][j] = (((double)rand() / RAND_MAX) * 6.0) - 3.0; // Random value between -3 and 3
            //nn->weights_to_hidden[i][j] = (((double)rand() / RAND_MAX) - 0.5) * 1.0; // Random value between -0.5 and 0.5
            nn->weights_to_hidden[i][j] = (((double)rand() / RAND_MAX) - 0.5) * 1.0; // Random value between -0.5 and 0.5
        }
    }

    for (int i = 0; i < nn->number_H - 1; i++) {
        for (int j = 0; j < nn->number_O; j++) {
            //nn->weights_to_output[i][j] = (((double)rand() / RAND_MAX) * 6.0) - 3.0; // Random value between -3 and 3
            //nn->weights_to_output[i][j] = (((double)rand() / RAND_MAX) - 0.5) * 1.0; // Random value between -0.5 and 0.5
            nn->weights_to_output[i][j] = (((double)rand() / RAND_MAX) - 0.5) * 1.0; // Random value between -0.5 and 0.5
        }
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

//Change this function to destroy the arrays related to the bias
void nn_destroy(NeuralNetwork *nn){
    if(nn){
        for (int i = 0; i < nn->number_I-1; i++) {
            free(nn->weights_to_hidden[i]);
        }
        free(nn->weights_to_hidden);

        for (int i = 0; i < nn->number_H-1; i++) {
            free(nn->weights_to_output[i]);
        }
        free(nn->weights_to_output);

        free(nn->I);
        free(nn->H);
        free(nn->O);
        free(nn->bias_to_output);
        free(nn->bias_to_hidden);
        free(nn->errors_output);
        free(nn->delta_hidden);
        free(nn->delta_output);
        free(nn);
    }
}

int main(int argc, char *argv[]){

    if (argc < 5) {
        printf("Usage: %s [iterations] [learning_rate] [threshold] [training_mode (0|1)]\n", argv[0]);
        return 1;
    }

    int maxIterations = atoi(argv[1]);
    double learningRate = atof(argv[2]);
    double threshold = atof(argv[3]); // Threshold to stop the training, 0.0001
    int xorMode = atoi(argv[4]); // 0 -> (normal training mode) | 1 - (XOR training mode)

    if(xorMode != 0 && xorMode != 1){
        printf("Usage: [training_mode] needs to be 0 (normal mode) or 1 (xor mode)\n");
        return 1;
    }

    if(xorMode == 0){
        NeuralNetwork *nn = nn_create(xorMode);
        double input_values[] = {2.0, 1.0};
        double output_values[] = {0.7, 0.4}; //Desired values, could be changed (between 0 and 1 because its a sigmoid function)

        //load_weights(nn);
        load_nn(nn);
        
        load_input_values(input_values, nn);

        double bias_weights_input_to_hidden[] = {2.0, 1.0};
        load_initial_bias_weights_to_hiddenLayer(bias_weights_input_to_hidden, nn);

        double bias_weights_hidden_to_output[] = {1.0, 2.0};
        load_initial_bias_weights_to_outputLayer(bias_weights_hidden_to_output, nn);

        int iteration = 0;
        
        while(iteration < maxIterations){
            printf("Iteration number %d\n", iteration);

            propagate(nn);

            //Calculate error of the outputs   
            for (int i = 0; i < nn->number_O; i++) {
                //printf("Value of Output[%d]: %lf\n",i, nn->O[i]);
                nn->errors_output[i] = output_values[i] - nn->O[i];
                //printf("Error[%d]: %lf\n",i, nn->errors_output[i]);
            }

            int number_outputs_within_threshold = 0;
            for(int i = 0; i < nn->number_O; i++){
                if(fabs(nn->errors_output[i]) <= threshold){
                    number_outputs_within_threshold++;
                }
            }

            //All output are within the threshold limit
            if(number_outputs_within_threshold == nn->number_O){
                printf("The output values are close to the desired values within the threshold lime at the iteration: %d\n", iteration);
                break;
            }

            //If still not valid output for each output unit, backpropagate
            back_propagate(nn, learningRate);

            iteration++;
        }
        for (int i = 0; i < nn->number_O; i++) {
            printf("Value of the final Output[%d]: %lf\n",i, nn->O[i]);
        }

        nn_destroy(nn);
    }else{
        double input_XOR_values[4][2] = {{0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}};
        double output_XOR_values[4] = {0.0, 1.0, 1.0, 0.0};
        //double trained_outputs[4];

        int all_outputs_within_threshold;
        int iteration = 0;

        NeuralNetwork *nn = nn_create(xorMode);
        load_weights(nn);
        
        double bias_weights_input_to_hidden[] = {0.3, 0.6}; //{2.0, 1.0}
        load_initial_bias_weights_to_hiddenLayer(bias_weights_input_to_hidden, nn);

        double bias_weights_hidden_to_output[] = {0.6, 0.3}; //{1.0, 2.0};
        load_initial_bias_weights_to_outputLayer(bias_weights_hidden_to_output, nn);

        FILE *outputFile;
        outputFile = fopen("output.txt", "w");

        if (outputFile == NULL) {
            printf("Error opening the output file.\n");
            return 1;
        }

        while(iteration < maxIterations){
            printf("Iteration number %d\n\n", iteration);

            all_outputs_within_threshold = 0;
            for(int i = 0; i < 4; i++){ // 4 vectors of input values
                load_input_values(input_XOR_values[i], nn);

                propagate(nn);

                //Calculate error of the outputs   
                for (int j = 0; j < nn->number_O; j++) {
                    //printf("Value of Output[%d]: %lf\n",i, nn->O[i]);
                    nn->errors_output[j] = output_XOR_values[i] - nn->O[j];
                    //printf("Error[%d]: %lf\n",i, nn->errors_output[i]);
                }

                for(int j = 0; j < nn->number_O; j++){
                    //printf("Error[%d] -> %lf <= %lf in sequence %d\n",j, fabs(nn->errors_output[j]), threshold, i);
                    if(fabs(nn->errors_output[j]) <= threshold ) {
                        all_outputs_within_threshold++;
                    }
                }
                printf("[%d,%d] - %lf --> %.2lf\n", (int)input_XOR_values[i][0], (int)input_XOR_values[i][1], nn->O[0], nn->O[0]);

                back_propagate(nn, learningRate);
            }
            fprintf(outputFile, "%d %d\n", iteration, all_outputs_within_threshold);

            if(all_outputs_within_threshold == 4){ //The outputs for each input vector are within the threshold limit
                printf("The output values are close to the desired values within the threshold lime at the iteration: %d\n", iteration);
                break;
            }
            iteration++;            
        }
        fclose(outputFile);
    }
    return 0;
}
