#include <iostream>
#include <time.h>
extern "C"
{
#include <immintrin.h>
}

using namespace std;

int main(){
const int col = 128, row = 64, num_trails = 100000;

float w[row][col];
float x[col];
double y[row];
float scratchpad[8];
for (int i=0; i<row; i++) {
    for (int j=0; j<col; j++) {
        w[i][j]=(float)(rand()%1000)/800.0f;
    }
}
for (int j=0; j<col; j++) {
    x[j]=(float)(rand()%1000)/800.0f;
}

__m256d sum_vec = _mm256_setzero_pd();

clock_t t1, t2;

t1 = clock();
for (int r = 0; r < num_trails; r++)
    for(int j = 0; j < row; j++)
    {
        __m256d wj = _mm256_load_pd((double*)&w[j][0]);
        __m256d x = _mm256_load_pd(&x[0]);
        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(wj, x));
    }
    _mm256_storeu_pd((double*)&y[0], sum_vec);
t2 = clock();
float diff = (((float)t2 - (float)t1) / CLOCKS_PER_SEC ) * 1000;
cout<<"With SIMD Time taken: "<<diff<<endl;
cout<< "Vector result: \n ";
    cout << "\t";
    for (int i=0; i<row; i++) {
        cout<<y[i]<<", ";
    }
    cout<<endl;
return 0;
}