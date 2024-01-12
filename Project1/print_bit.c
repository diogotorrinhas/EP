#include <stdio.h>


int main(){
    int i;
    int p;

    printf("Enter a number (i): ");
    scanf("%d", &i);

    printf("Enter a bit position (p): ");
    scanf("%d", &p);

    printf("The bit at position %d of number %d is %d\n", p, i, (i >> p) & 1);

    // 10[00001010] >>3 [00000001]
    // 5 [00000101] >>2 [00000001]
    // 12[00001100] >>2 [00000011]

    return 0;
}