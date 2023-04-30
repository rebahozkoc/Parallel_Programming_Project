//compile with : g++ FIXED_mat_mult_avx.cpp  -std=c++11 -O3 -march=native
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
  float y[row];
  
  for (int i=0; i<row; i++) {
    for (int j=0; j<col; j++) {
      w[i][j]=(float)(rand()%1000)/800.0f;
    }
  }
  for (int j=0; j<col; j++) {
    x[j]=(float)(rand()%1000)/800.0f;
  }

  // Fix: Allocate y array in aligned memory
  float *y_aligned = (float*) _mm_malloc(row * sizeof(float), 32);
 
  clock_t t1, t2;
 
  t1 = clock();
  for (int r = 0; r < num_trails; r++)
    for(int j = 0; j < row; j++)
      {
	float sum = 0;
	__m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
	float *wj = w[j];
	ymm0 = _mm256_load_ps(x);
	ymm4 = _mm256_load_ps(wj);
	ymm1 = _mm256_permute2f128_ps(ymm4, ymm4, 1);
	ymm5 = _mm256_permute2f128_ps(ymm0, ymm0, 1);
	ymm2 = _mm256_mul_ps(ymm0, ymm4);
	ymm6 = _mm256_mul_ps(ymm1, ymm5);
	ymm3 = _mm256_hadd_ps(ymm2, ymm6);
	ymm7 = _mm256_permute_ps(ymm3, 0x4E);
	ymm3 = _mm256_add_ps(ymm3, ymm7);
	ymm7 = _mm256_permute_ps(ymm3, 0xB1);
	ymm3 = _mm256_add_ps(ymm3, ymm7);
	_mm256_store_ps(&y_aligned[j], ymm3);
      }
  t2 = clock();
  float diff = (((float)t2 - (float)t1) / CLOCKS_PER_SEC ) * 1000;
  cout<<"With SIMD Time taken: "<<diff<<endl;
  cout<< "Vector result: \n ";
  cout << "\t";
  for (int i=0; i<row; i++) {
    cout<<y_aligned[i]<<", ";
  }
  cout<<endl;

  // Fix: Free aligned memory
  _mm_free(y_aligned);

  return 0;
}
