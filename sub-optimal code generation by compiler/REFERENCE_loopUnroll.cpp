//Compile with g++ REFERENCE_loopUnroll.cpp
#include <iostream>
#include <chrono>

using namespace std;

#define N 256*256*256

void sum_unrolled_prefetched(int* a, int* b ,int &sum) {


  for (int i = 0; i < N-4; i += 4) {
    __builtin_prefetch(&a[i + 8]);
    __builtin_prefetch(&b[i + 8]);
    sum += a[i] * b[i];
    sum += a[i + 1] * b[i + 1];
    sum += a[i + 2] * b[i + 2];
    sum += a[i + 3] * b[i + 3];
  }
  for (int i = N - 4; i < N; i++)
    sum += a[i] * b[i];
}

int main() {
  cout << "An integer is " << sizeof(int) << " bytes " << endl;

  int* a = new int[N];
  int* b = new int[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }

  int sum=0;
  auto t1 = std::chrono::high_resolution_clock::now();
  sum_unrolled_prefetched(a, b, sum);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << "sum_unrolled_prefetched: " << sum << " " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";

  return 0;
}
