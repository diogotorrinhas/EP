#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <ctype.h>
#include <math.h>
#include "TicTacToe_NeuralNetwork.h"

#define BOARD_SIZE 3

char board[BOARD_SIZE][BOARD_SIZE];

NeuralNetwork* nn_create(){
    int number_inputs, number_hidden, number_outputs;
    NeuralNetwork *nn = malloc(sizeof(NeuralNetwork));
    if (nn == NULL) {
        fprintf(stderr, "Error allocating memory for NeuralNetwork\n");
        exit(EXIT_FAILURE);
    }
    
    FILE *fp = fopen("TicTac_NeuralNetwork.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file TicTac_NeuralNetwork.txt\n");
        exit(EXIT_FAILURE); 
    }

    // Read the first line (I H O)
    if (fscanf(fp, "%d %d %d", &number_inputs, &number_hidden, &number_outputs) == 3) {
        nn->number_I = number_inputs + 1; //Add one bias unit to the last position of the input layer
        nn->number_H = number_hidden + 1; //Add one bias unit to the last position of the hidden layer
        nn->number_O = number_outputs;
        fclose(fp);
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
    FILE *fp = fopen("Trained_Network.txt", "r");
    int fl,sl, input_unit, hidden_unit;
    double weight;
    int unit; // unit = hidden or output and is the unit that is connected to the bias
    double bias_weight;

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (sscanf(line, "%d:%d %d:%d %lf", &fl, &input_unit, &sl, &hidden_unit, &weight) == 5) {
            if (fl == 1) {
                nn->weights_to_hidden[input_unit - 1][hidden_unit - 1] = weight;
            } else if (fl == 2) {
                nn->weights_to_output[input_unit - 1][hidden_unit - 1] = weight;
            }
        } else if (sscanf(line, "%d:%d %lf", &fl, &unit, &bias_weight) == 3) {
            if (fl == 1) {
                nn->bias_to_hidden[unit - 1] = bias_weight;
            } else if (fl == 2) {
                nn->bias_to_output[unit - 1] = bias_weight;
            }
            //printf("Layer: %d, Unit: %d, Weight: %lf\n", fl, unit, bias_weight);
        }
    }
    fclose(fp);
}


void write_nn(NeuralNetwork *nn){
    FILE *fp = fopen("Trained_Network.txt", "w");

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // fprintf(fp, "%d %d %d\n", nn->number_I, nn->number_H, nn->number_O);

    for (int i = 0; i < nn->number_I-1; i++) {
        for (int j = 0; j < nn->number_H-1; j++) {
            fprintf(fp, "1:%d 2:%d %lf\n", i + 1, j + 1, nn->weights_to_hidden[i][j]);
        }
    }

    for (int i = 0; i < nn->number_H-1; i++) {
        for (int j = 0; j < nn->number_O; j++) {
            fprintf(fp, "2:%d 3:%d %lf\n", i + 1, j + 1, nn->weights_to_output[i][j]);
        }
    }

    //Write the bias
    for (int i = 0; i < nn->number_H-1; i++) {
        fprintf(fp, "1:%d %lf\n", i + 1, nn->bias_to_hidden[i]);
    }

    for(int i = 0; i < nn->number_O; i++){
        fprintf(fp, "2:%d %lf\n", i + 1, nn->bias_to_output[i]);
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


//Initialize the board with empty spaces
void initializeBoard() {
    for(int i = 0; i < BOARD_SIZE; i++){
        for(int j = 0; j < BOARD_SIZE; j++){
            board[i][j] = ' '; // ' ' = empty
        }
    }
}

// Function to print the current state of the game board
void printBoard() {
    printf("\n");
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            printf(" %c ", board[i][j]);
            if (j < BOARD_SIZE - 1){
                printf("|");
            } 
        }
        printf("\n");
        if (i < BOARD_SIZE - 1) {
            printf("-----------\n");
        }
    }
    printf("\n");
}

int isValidMove(int row, int col){
    if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE && board[row][col] == ' '){
        return 1;
    }else{
        return 0;
    }
}


// Function to get the indices of empty positions
int* getEmptyPositions(int* count) {
    int* emptyPositions = (int*)malloc(BOARD_SIZE * BOARD_SIZE * sizeof(int));
    *count = 0;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (board[i][j] == ' ') {
                emptyPositions[(*count)++] = i * BOARD_SIZE + j;
            }
        }
    }
    return emptyPositions;
}

int getEmptyPositionsCount() {
    int count = 0;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (board[i][j] == ' ') {
                count++;
            }
        }
    }
    return count;
}

//check if the game is over without winner, i.e, tie.
int checkTie(){
    int tie = 1;
    for(int i = 0; i < BOARD_SIZE; i++){
        for(int j = 0; j < BOARD_SIZE; j++){
            if(board[i][j] == ' '){
                tie = 0;
                break;
            }
        }
    }
    return tie;
}

//Check if there is a winner
int checkWinner(char player){
    int winner = 0;
    //check rows
    for(int i = 0; i < BOARD_SIZE; i++){
        if(board[i][0] == player && board[i][1] == player && board[i][2] == player){
            winner = 1;
            return winner;
        }
    }
    //check columns
    for(int i = 0; i < BOARD_SIZE; i++){
        if(board[0][i] == player && board[1][i] == player && board[2][i] == player){
            winner = 1;
            return winner;
        }
    }
    //check diagonals
    if(board[0][0] == player && board[1][1] == player && board[2][2] == player){
        winner = 1;
        return winner;
    }
    if(board[0][2] == player && board[1][1] == player && board[2][0] == player){
        winner = 1;
        return winner;
    }
    return winner;
}

int get_higher_value_pos(NeuralNetwork *nn, int *pos, int number_iterations){
    int higher_value_index = -1;
    //double max = nn->O[0];
    double max = 0.0;

    for (int i = 0; i < nn->number_O; i++) {
        int skip = 0;
        // Check if the current index i is in the pos array
        for (int j = 0; j < number_iterations; j++) {
            if (i == pos[j]) {
                skip = 1;
                break;
            }
        }
        // Skip this iteration if i is in the pos array
        if (skip) {
            continue;
        }
        // Find the maximum value among non-excluded indices
        if (nn->O[i] > max) {
            max = nn->O[i];
            higher_value_index = i;
        }
    }
    return higher_value_index;
}

int bot_random_generator_move(char choice){
    char currentPlayer = '\0';
    if (choice == 'n'){
        currentPlayer = 'X';
    }else if(choice == 'y'){
        currentPlayer = 'O';
    }

    int win = 0;

    int count;
    int* emptyPositions = getEmptyPositions(&count);
    if(count != 0){
        int randomIndex = rand() % count;
        int position = emptyPositions[randomIndex];
        int row = position / BOARD_SIZE;
        int col = position % BOARD_SIZE;
        board[row][col] = currentPlayer;
        printf("\n");
        printf("Bot move (%c): %d, %d\n", currentPlayer, row, col);
    }

    if (checkWinner(currentPlayer)) {
        printf("Player %c wins!\n", currentPlayer);
        win = 1;
    }
    free(emptyPositions);
    return win;
}

int agent_move(NeuralNetwork *nn, char choice){
    char currentPlayer;
    if (choice == 'n'){
        currentPlayer = 'X';
    }else if(choice == 'y'){
        currentPlayer = 'O';
    }

    int win = 0;
    int count;
    int *emptyPositions = getEmptyPositions(&count);
    //int index_higher = -1;
    int *index_higher = (int*)malloc(9 * sizeof(int));
    if (index_higher == NULL) {
        fprintf(stderr, "Error allocating memory for index_higher variable\n");
        exit(EXIT_FAILURE);
    }

    int empty_pos = 0;
    int higher_pos;
    int number_iterations = 0;

    if(count > 1){ //count != 0
        while(1){
            //Obtain the index of the higher value in the output layer
            higher_pos = get_higher_value_pos(nn, index_higher, number_iterations);
            //printf("index_higher: %d \n", higher_pos);

            for(int i = 0; i < count; i++){
                // Check if the higher value index is an empty position
                if(higher_pos == emptyPositions[i]){
                    empty_pos = 1;
                    break;
                }
            }

            if(empty_pos == 1){
                break;
            }else if(number_iterations == 9){ // 8 means that index = 8, which means that the size of the array is 9. Index 0,1...8
                break;
            }else{
                index_higher[number_iterations++] = higher_pos;
            } 
        }

        if(number_iterations == 9){
            int randomIndex = rand() % count;
            higher_pos = emptyPositions[randomIndex];
        }
        int row = higher_pos / BOARD_SIZE;
        int col = higher_pos % BOARD_SIZE;
        board[row][col] = currentPlayer;
        printf("\n");
        printf("Bot move (%c): %d, %d\n", currentPlayer, row, col);
        if (checkWinner(currentPlayer)) {
            printf("Player %c wins!\n", currentPlayer);
            win = 1;
        }   
    }else if(count == 1){
        int row = emptyPositions[0] / BOARD_SIZE;
        int col = emptyPositions[0] % BOARD_SIZE;
        board[row][col] = currentPlayer;
        printf("\n");
        printf("Bot move (%c): %d, %d\n", currentPlayer, row, col);
        if (checkWinner(currentPlayer)) {
            printf("Player %c wins!\n", currentPlayer);
            win = 1;
        }
    }
    free(index_higher);
    free(emptyPositions);
    return win;
}

bool isNumber(char number[])
{
    int i = 0;

    //checking for negative numbers
    if (number[0] == '-')
        i = 1;
    for (; number[i] != 0; i++)
    {
        if (!isdigit(number[i]))
            return false;
    }
    return true;
}


void predefinedBoardState(char *boardState){
    int k = 0;

    // Check if the length of the board configuration is valid
    if (strlen(boardState) != BOARD_SIZE * BOARD_SIZE) {
        fprintf(stderr, "Invalid board configuration size.\n");
        exit(1);
    }

    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            char currentChar = boardState[k++];
            
            // Check if the character is a valid symbol (X, O, or -)
            if (currentChar != 'X' && currentChar != 'O' && currentChar != '-') {
                fprintf(stderr, "Invalid character in board configuration: %c\n", currentChar);
                exit(1);
            }

            if(currentChar != '-'){
                board[i][j] = currentChar;
            }else{
                board[i][j] = ' ';
            }
        }
    }
}

void writeBoardToFile(FILE *file) {
    for (int i = 0; i < BOARD_SIZE; i++) {
        for (int j = 0; j < BOARD_SIZE; j++) {
            if(board[i][j] == ' '){
                fprintf(file, "%c", '-');
            }else{
                fprintf(file, "%c", board[i][j]);
            }
        }
    }
    fprintf(file, " ");
}

void inputStringToNeuralNetwork(char *boardState, double *inputs) {
    int index = 0;

    for (int i = 0; i < 9; i++) {
        if (boardState[i] == 'X') {
            inputs[index++] = 1;
        } else if (boardState[i] == 'O') {
            inputs[index++] = -1;
        } else {
            inputs[index++] = 0;
        }
    }
}

void giveDiseredOutputForNeuralNetwork(char *boardState, char *boardNextState, double *desired_outputs){
    int index = 0;

    if(boardNextState[0] == '\0'){
        for(int i = 0; i < 10; i++){
            desired_outputs[index++] = 0;
        }
    }else{
        for(int i = 0; i < 9; i++){
            if(boardState[i] == boardNextState[i]){
                desired_outputs[index++] = 0;
            }else if(boardState[i] != boardNextState[i]){
                desired_outputs[index++] = 1;
            }
        }
    } 
}

int checkIfLastBoardStateForCurrentGame(char *boardState){
    int count = 0;

    for(int i = 0; i < 9; i++){
        if(boardState[i] == '-'){
            count++;
        }
    }

    if(count == 9){
        return 1;
    }else{
        return 0;
    }
}

char *getCurrentBoardState(){
    char *boardState = malloc(10 * sizeof(char));
    int index = 0;

    for(int i = 0; i < BOARD_SIZE; i++){
        for(int j = 0; j < BOARD_SIZE; j++){
            if(board[i][j] == ' '){
                boardState[index++] = '-';
            }else{
                boardState[index++] = board[i][j];
            }
        }
    }
    boardState[index] = '\0';
    return boardState;
}


int main(int argc, char *argv[]){

    if ((strcmp(argv[1], "m") == 0 || strcmp(argv[1], "b") == 0 || strcmp(argv[1], "a") == 0 || strcmp(argv[1], "t") == 0) && argc >= 2) {

        //No board configuration provided, for example, ./TicTacToe_Game b 5
        if (argc == 3 && isNumber(argv[2]) == 1) { 
            srand(atoi(argv[2]));
            initializeBoard();

        //Board configuration provided and no seed, for example, ./TicTacToe_Game b XOX-X-XOX or ./TicTacToe_Game m XOX-X-XOX or ./TicTacToe_Game a XOX-X-XOX  
        }else if(argc == 3 && isNumber(argv[2]) == 0){
            srand(time(NULL));
            predefinedBoardState(argv[2]);

        } else if (argc == 4) {
            // If the seed is provided, board configuration too, for example, ./TicTacToe_Game b 5 XOX-X-XOX
            predefinedBoardState(argv[3]);
            srand(atoi(argv[2]));

        } else {
            srand(time(NULL));
            initializeBoard(); 
        }
    } else {
        printf("Usage: %s [game_mode (m|b|a|t)] [seed (optional)] [board (optional)]\n", argv[0]); 
        return 1;
    }

    char game_mode  = argv[1][0];
    char currentPlayer = 'X';
    int row, col;

    //Auxiliar variable to write the state board in file games.txt
    int count;
    
    if(game_mode == 'm'){ //EX1
        //Esta escrita vai sair, serve apenas para introduzir jogos no ficheiro
        FILE *fp;
        fp = fopen("games.txt", "a");

        if (fp == NULL) {
            fprintf(stderr, "Error opening the file.\n");
            return 1;
        }

        while(1){
            printBoard();

            count = getEmptyPositionsCount();
            if (count == 9) {
                // write the first state of the board
                writeBoardToFile(fp); 
            }

            printf("Player %c, enter row and column (e.g, 0 0 or 0 1): ", currentPlayer);
            scanf("%d %d", &row, &col);

            // Validate the move
            if (!isValidMove(row, col)) {
                printf("Invalid move! Try again.\n");
                continue;
            }

            board[row][col] = currentPlayer;
            writeBoardToFile(fp); //write the state of the board to the games.txt file

            // Check if the game is over (winner or tie)
            if (checkWinner(currentPlayer)) {
                printBoard();
                printf("Player %c wins!\n", currentPlayer);
                break;
            }else if(checkTie()){
                printBoard();
                printf("Tie!\n");
                break;
            }

            if(currentPlayer == 'X'){
                currentPlayer = 'O';
            }else{
                currentPlayer = 'X';
            }
        }

        fprintf(fp, "\n");
        fclose(fp);

    }else if(game_mode == 'b'){ //Random agent EX2
        int bot_win = 0;
        char choice;
        printf("Do you want to play first? (y/n):\n");
        scanf("%c", &choice);

        switch (choice)
        {
        case 'y':
            while(1){
                printBoard();
                
                printf("Player %c, enter row and column (e.g, 0 0 or 0 1): ", currentPlayer);
                scanf("%d %d", &row, &col);

                // Validate the move
                if (!isValidMove(row, col)) {
                    printf("Invalid move! Try again.\n");
                    continue;
                }

                board[row][col] = currentPlayer;

                // Check if the game is over (winner or tie)
                if (checkWinner(currentPlayer)) {
                    printBoard();
                    printf("Player %c wins!\n", currentPlayer);
                    break;
                }else if(checkTie()){
                    printBoard();
                    printf("Tie!\n");
                    break;
                }

                printBoard();

                bot_win = bot_random_generator_move(choice);
                if(bot_win){
                    printBoard();
                    break;
                }else if(checkTie()){
                    printBoard();
                    printf("Tie!\n");
                    break;
                }
            }
            break;
        case 'n':
            while(1){
                currentPlayer = 'O';
                bot_win = bot_random_generator_move(choice);
                if(bot_win){
                    printBoard();
                    break;
                }else if(checkTie()){
                    printBoard();
                    printf("Tie!\n");
                    break;
                }
                printBoard();

                int validMove = 0;
                do {
                    printf("Player %c, enter row and column (e.g., 0 0 or 0 1): ", currentPlayer);
                    scanf("%d %d", &row, &col);

                    // Validate the move
                    validMove = isValidMove(row, col);

                    if (!validMove) {
                        printf("Invalid move! Try again.\n");
                    }
                } while (!validMove);

                board[row][col] = currentPlayer;

                // Check if the game is over (winner or tie)
                if (checkWinner(currentPlayer)) {
                    printBoard();
                    printf("Player %c wins!\n", currentPlayer);
                    break;
                }else if(checkTie()){
                    printBoard();
                    printf("Tie!\n");
                    break;
                }

                printBoard();
            }
            break;
        default:
            printf("Invalid choice!\n");
            break;
        }
    }else if(game_mode == 'a'){ //EX3
        NeuralNetwork *nn = nn_create();
        load_nn(nn);

        char choice;
        printf("Do you want to play first? (y/n):\n");
        scanf("%c", &choice);
        
        int agent_win = 0;
        double current_inputs[9];
        char *currentBoardState = NULL;

        switch(choice){
            case 'y':
                while(1){
                    printBoard();

                    printf("Player %c, enter row and column (e.g, 0 0 or 0 1): ", currentPlayer);
                    scanf("%d %d", &row, &col);

                    // Validate the move
                    if (!isValidMove(row, col)) {
                        printf("Invalid move! Try again.\n");
                        continue;
                    }

                    board[row][col] = currentPlayer;

                    // Check if the game is over (winner or tie)
                    if (checkWinner(currentPlayer)) {
                        printBoard();
                        printf("Player %c wins!\n", currentPlayer);
                        break;
                    }else if(checkTie()){
                        printBoard();
                        printf("Tie!\n");
                        break;
                    }

                    printBoard();

                    currentBoardState = getCurrentBoardState();
                    //printf("Current Board State: %s\n", currentBoardState);
                    inputStringToNeuralNetwork(currentBoardState, current_inputs);
                    load_input_values(current_inputs, nn);
                    
                    propagate(nn);

                    agent_win = agent_move(nn, choice);
                    if(agent_win){
                        printBoard();
                        break;
                    }else if(checkTie()){
                        printBoard();
                        printf("Tie!\n");
                        break;
                    }
                }
                break;
            case 'n':
                currentPlayer = 'O';
                while(1){
                    currentBoardState = getCurrentBoardState();
                    //printf("Current Board State: %s\n", currentBoardState);
                    inputStringToNeuralNetwork(currentBoardState, current_inputs);
                    load_input_values(current_inputs, nn);
                    
                    propagate(nn);

                    agent_win = agent_move(nn, choice);
                    if(agent_win){
                        printBoard();
                        break;
                    }else if(checkTie()){
                        printBoard();
                        printf("Tie!\n");
                        break;
                    }

                    printBoard();

                    int validMove = 0;
                    do {
                        printf("Player %c, enter row and column (e.g., 0 0 or 0 1): ", currentPlayer);
                        scanf("%d %d", &row, &col);

                        // Validate the move
                        validMove = isValidMove(row, col);

                        if (!validMove) {
                            printf("Invalid move! Try again.\n");
                        }
                    } while (!validMove);

                    board[row][col] = currentPlayer;

                    // Check if the game is over (winner or tie)
                    if (checkWinner(currentPlayer)) {
                        printBoard();
                        printf("Player %c wins!\n", currentPlayer);
                        break;
                    }else if(checkTie()){
                        printBoard();
                        printf("Tie!\n");
                        break;
                    }

                    printBoard();
                }
                break;

            default:
                printf("Invalid choice!\n");
                break;
        }
        free(currentBoardState);
        nn_destroy(nn);
    }else if(game_mode == 't'){ //Training neural network
        FILE *fp;
        fp = fopen("games.txt", "r");

        if (fp == NULL) {
            fprintf(stderr, "Error opening the file.\n");
            return 1;
        }

        char *boardStates[1000]; //1000 board states, default size, which means, max 1000 board states in the file
        
        // Allocate memory for each board state
        for (int i = 0; i < 1000; i++) {
            boardStates[i] = malloc(10); // 9 + '\0' = 10
            if (boardStates[i] == NULL) {
                fprintf(stderr, "Memory allocation error.\n");
                // Handle error and free allocated memory
                for (int j = 0; j < i; j++) {
                    free(boardStates[j]);
                }
                fclose(fp);
                return 1;
            }
        }
        
        int numberOfBoardStates = 0;
        while(numberOfBoardStates < 1000 && fscanf(fp, "%9s", boardStates[numberOfBoardStates]) != EOF){
            numberOfBoardStates++;
        }
        fclose(fp);
        
        int maxIterations = 150000;  //100000
        double learningRate = 0.7; // >= 0.5
        double threshold = 0.01; // 0.01
        int all_outputs_within_threshold;
        int iteration = 0;

        // Train the neural network
        NeuralNetwork *nn = nn_create();
        load_weights(nn);

        double bias_weights_input_to_hidden[] = {0.3, 0.6}; //{2.0, 1.0}
        load_initial_bias_weights_to_hiddenLayer(bias_weights_input_to_hidden, nn);
        double bias_weights_hidden_to_output[] = {0.6, 0.3}; //{1.0, 2.0};
        load_initial_bias_weights_to_outputLayer(bias_weights_hidden_to_output, nn);

        double inputs[9];
        double desired_outputs[9];

        printf("Starting training the neural network...\n");
        while(iteration < maxIterations){
            all_outputs_within_threshold = 0;

            //Iterate over the board states
            for(int i = 0; i < numberOfBoardStates; i++){
                if(checkIfLastBoardStateForCurrentGame(boardStates[i+1]) == 0){
                    inputStringToNeuralNetwork(boardStates[i], inputs);
                    giveDiseredOutputForNeuralNetwork(boardStates[i], boardStates[i+1], desired_outputs);

                    load_input_values(inputs, nn);
                    propagate(nn);

                    printf("BoardState at iteration %d: %s\n", iteration, boardStates[i]);
                    //Calculate error of the outputs   
                    for (int j = 0; j < nn->number_O; j++) { // number_O is 9
                        printf("Input[%d]: %lf | Desired output[%d]: %lf",j, inputs[j], j, desired_outputs[j]);
                        nn->errors_output[j] = desired_outputs[j] - nn->O[j];
                        printf(" Value of Output [%d]: %lf | Error[%d]: %lf\n",j, nn->O[j], j, nn->errors_output[j]);
                    }
                    printf("--------------------\n");

                    for(int j = 0; j < nn->number_O; j++){
                        if(fabs(nn->errors_output[j]) <= threshold ) {
                            all_outputs_within_threshold++;
                        }
                    }
                    back_propagate(nn, learningRate);
                }
            }
            
            if(all_outputs_within_threshold == numberOfBoardStates * 9){
                printf("All outputs within threshold! Train finished at iteration %d !\n", iteration);
                break;
            }
            iteration++;
        }
        printf("Training finished!\n");
        write_nn(nn);
        nn_destroy(nn);
    }
    return 0;
}