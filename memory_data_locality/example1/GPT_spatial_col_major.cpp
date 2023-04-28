#include<iostream>
#include<ctime>
#include<omp.h>
#include <iomanip>

using namespace std;

#define N 10000

int main () {
    int i, j;
    double sum;
    clock_t start, end;

    double* b = new double[N*N];
    for (i = 0; i < N; i++) { 
        for (j = 0; j < N; j++) {
            b[i*N + j] = i * j + 0.0f;
        }
    }

    // Transpose the matrix
    for (i = 0; i < N; i++) {
        for (j = i+1; j < N; j++) {
            double temp = b[i*N + j];
            b[i*N + j] = b[j*N + i];
            b[j*N + i] = temp;
        }
    }

    double* column_sum = new double[N];  
    for (i = 0; i < N; i++) { 
        column_sum[i] = 0.0;
    }

    start = clock();
    #pragma omp parallel for private(j,sum)
    for (i = 0; i < N; i++) {
        sum = 0;
        for (j = 0; j < N; j++) {
            sum += b[i*N + j];
        }
        column_sum[i] = sum;
    }
    end = clock();

    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by program is : " << fixed
         << time_taken << setprecision(5);
    cout << " sec " << endl;

    delete [] b;
    delete [] column_sum;
    return 0;
}
