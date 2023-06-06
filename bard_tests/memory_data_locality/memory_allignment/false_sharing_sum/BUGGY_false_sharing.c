    #include <omp.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h> 

    const int multiplier = 32;
    int main() { 
    unsigned int N = 1 << 30;
    int* num_to_add = (int*) malloc(sizeof(int) * N); 
    for(int i = 0; i < N; i++) num_to_add[i] = 1;

    int *thread_shared =  (int*) malloc(sizeof(int) * multiplier * 32);
    
    printf("starting experiments\n");
    for(int t = 2; t <= 2; t <<= 1) {
        memset(thread_shared, 0, sizeof(int) * multiplier * 32);      

        double st = omp_get_wtime();
    #pragma omp parallel num_threads(t) 
        {
        int tid = omp_get_thread_num();
    #pragma omp for 
        for(int i = 0; i < N; i++) {
        thread_shared[tid * multiplier] += num_to_add[i];
        }
        }
        double tm = omp_get_wtime() - st;
        
        int sum = 0;
        for(int i = 0; i < t; i++) {
        sum += thread_shared[i * multiplier];
        }
        printf("%d threads - %d sum - in %f seconds\n", t, sum, tm);
    }
    }