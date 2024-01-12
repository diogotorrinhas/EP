#include <stdio.h>

int main(){
    int i;
    
    printf("Input a number: ");
    scanf("%d", &i);

    int numBits = sizeof(i) * 8;

    printf("Result: \n");
    for(int j = numBits - 1; j >= 0; j--){
        printf("%d", (i >> j) & 1);
    }

    // [00000000 00000000 00000000 00000101] -> 5 (i) >>32 = [00000000 00000000 00000000 00000000 & 1 -> 00000000 00000000 00000000 0000000(0)]...

    printf("\n");
    return 0;
}