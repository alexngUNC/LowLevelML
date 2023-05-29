#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define train_count (sizeof(train)/sizeof(train[0]))
#define EPOCHS 1000

// Predicts number based on input numbers (pairs)
float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

float rand_float(void) {
    return (float) rand() / (float) RAND_MAX;
}

float cost(float w) {
    // MSE
    float loss = 0.0f;

    for (size_t i=0; i<train_count; i++) {
        
        // Get input number (next example in data)
        float x = train[i][0];

        // Output is input multiplied by weight
        float y = x*w;
        
        // Error
        float e = y - train[i][1];
        loss += e*e;
    }
    
    // MSE - get the average of summed squared errors
    loss /= train_count;
    return loss;

}

int main() {
    // Start with random parameter between 0 and 1
    // Seed
    srand(69);
    
    // srand(time(0));
    float w = rand_float() * 10.0f;
    float eps = 1e-3;
    float rate = 1e-3;

    for (int i=0; i<EPOCHS; i++) {
        // Approximited derivative with finite difference
        float dcost = (cost(w + eps) - cost(w))/eps;
        printf("Epoch %d: | Loss: %f\n", i, cost(w));

        // Update weights with learning rates
        w -= rate*dcost;
    }
    printf("Approximate function: y = x*%f\n", w);    
    return 0;
}
