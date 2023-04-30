#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <chrono>
using namespace std;

#define N 1024 * 1024 * 1024

void dot_norm(int* a, int* b, int& sum) {
  __m128i vec_sum = _mm_setzero_si128();
  for (int i = 0; i < N; i+=4){
    __m128i vec_a = _mm_load_si128((__m128i *)&a[i]);
    __m128i vec_b = _mm_load_si128((__m128i *)&b[i]);
    vec_sum = _mm_add_epi32(vec_sum, _mm_mul_epu32(vec_a, vec_b));
  }
  sum = _mm_extract_epi32(vec_sum, 0) + _mm_extract_epi32(vec_sum, 1) + _mm_extract_epi32(vec_sum, 2) + _mm_extract_epi32(vec_sum, 3);
}

int main() {
  cout << "An int is " << sizeof(int) << " bytes "<< endl;

  int* a = (int*)_mm_malloc(N * sizeof(int), 16); 
  int* b = (int*)_mm_malloc(N * sizeof(int), 16);
  
  for(int i = 0; i < N; i++) {
    a[i] = i % 0x000000FF;
    b[i] = i % 0x000000FF;
  }
  int sum = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  dot_norm(a, b, sum);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << sum << " dot_norm " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";
  return 0;
}
