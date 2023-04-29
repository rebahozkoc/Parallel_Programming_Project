//compile with: g++ prefetch.cpp  -std=c++11 -O3 
//use -fopt-info-vec-optimized for vectorization info verbose
//run: ./a.out

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
using namespace std;

#define N 1024 * 1024 * 1024

void dot_norm(int* a, int* b, int& sum) {
  sum = 0;
  for (int i = 0; i < N; i++){
    sum += a[i]*b[i];
  }
}

void dot_simd(int* a, int* b, int& sum) {
  __m128i *ap = (__m128i*)a, *bp = (__m128i*)b;
  __m128i s; for(int i = 0; i < 4; i++) s[i] = 0;

  //assume N is divisible by 4
  sum = 0;
  for (int i = 0; i < N; i += 4) {
    /*    uint32_t *val = (uint32_t*)ap;
    uint32_t *val2 = (uint32_t*)bp;
    cout << val[0] << " ---  " << val2[0] << endl;
    cout << val[1] << " ---  " << val2[1] << endl;
    cout << val[2] << " ---  " << val2[2] << endl;
    cout << val[3] << " ---  " << val2[3] << endl;*/
    __m128i temp = _mm_mullo_epi32(*(ap++), *(bp++));
    //    uint32_t *val3 = (uint32_t*)&temp;
    //cout << val3[0] << " " << val3[1] << " " << val3[2] << " " << val3[3] << endl;
    s = _mm_add_epi32(temp, s);
    //uint32_t *val4 = (uint32_t*)&s;
    //cout << val4[0] << " " << val4[1] << " " << val4[2] << " " << val4[3] << endl;
    //  if(i > 20) exit(1);
  }
  uint32_t *val = (uint32_t*)&s;
  sum = 0; for(int i = 0; i < 4; i++) sum += val[i];
}

int main() {
  cout << "An int is " << sizeof(int) << " bytes "<< endl;

  int* a = (int*)_mm_malloc(N * sizeof(int), 16); 
  int* b = (int*)_mm_malloc(N * sizeof(int), 16);
  
  for(int i = 0; i < N; i++) {
    a[i] = i % 0x000000FF;
    b[i] = i % 0x000000FF;
  }
  cout << a << " " << b << endl;
  
  int sum = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  dot_norm(a, b, sum);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " dot_norm " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";

  sum = 0;
  t1 = std::chrono::high_resolution_clock::now();
  dot_simd(a, b, sum);
  t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " dot_simd " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";
  return 0;
}
