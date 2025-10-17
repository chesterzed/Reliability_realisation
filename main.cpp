//
// Created by chest on 07.10.2025.
//

#include "distributions/beta/beta_dist.h"
//#include <iostream>

int main(void) {
//    printf("Hello world");
    BetaDistribution diagram = BetaDistribution(3, 5);
    diagram.plot(100, true);
    for (int i = 0; i <= 10; ++i) {
        printf("\n%lf", diagram.pdf((double)i/10));
        printf(" - %lf", (double)i/10);
    }
    // Подготовка данных


    return 0;
}