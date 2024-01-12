#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h> 

#define NUMBER_BITS 32

//Usage(Example): ./bits dtb 4 -> Resultado: 00000000000000000000000000000100
//Compile: gcc -o bits bits.c -lm

int bin_to_int(char bits[]){
    int num = 0;
    for(int i = NUMBER_BITS-1; i >= 0; i--){
        num += pow(2,31-i)*(bits[i]-48);
    }
    return num;
}

int main(int argc, char *argv[]){
    int i;
    char bits[NUMBER_BITS];
    
    if(argc != 3){
        printf("Usage: ./bits <dtb|btd> <number(32 bits if binary)>\n");
        return 1;
    }

    char* number = argv[2];
    for(int i = 0; i < number[i] != '\0'; i++){
        if(!isdigit(number[i])){
            printf("The last Argument is not a valid number\n");
            return 1;
        }
    }

    if(strcmp(argv[1],"dtb") == 0){
        //Decimal to binary
        int i = atoi(argv[2]);
        int numBits = sizeof(i) * 8; //integer is 4 bytes-> 4*8 = 32 bits

        printf("Decimal to bin Result: \n");
        for(int j = numBits - 1; j >= 0; j--){
            printf("%d", (i >> j) & 1);
        }
        printf("\n");
    }else if(strcmp(argv[1],"btd") == 0){
        //Binary to decimal
        char* bits = argv[2];
        int num = bin_to_int(bits);
        printf("Binary to decimal Result: %d\n", num);
    }
    
    return 0;
}