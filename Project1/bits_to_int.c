#include <stdio.h>
#include <math.h>

#define NUMBER_BITS 32

//tested with 32 bits (integer)
// 00000000000000000000000000000101 -> 5
// 00000000000000000000000000000111 -> 7
// 00000000000000000000000000001011 -> 11


int main(){
    char bits[NUMBER_BITS];
    
    printf("Input a string of binary digits (32): ");
    scanf("%s", bits);

    int num = 0;
    for(int i = NUMBER_BITS-1; i >= 0; i--){
        num += pow(2,31-i)*(bits[i]-48);
    }

    printf("Result: %d\n", num);
    return 0;
}