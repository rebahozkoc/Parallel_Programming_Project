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
