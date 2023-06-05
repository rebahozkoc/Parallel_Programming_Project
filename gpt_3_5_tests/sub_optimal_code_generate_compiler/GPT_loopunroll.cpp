#include <iostream>
#include <chrono>

using namespace std;

#define N 256*256*256

void sum_normal(int* a, int* b, int& sum) {
  sum = 0;

  for (int i = 0; i < N - 7; i += 8) {
    sum += (a[i] * b[i]) + (a[i + 1] * b[i + 1]) + (a[i + 2] * b[i + 2]) + (a[i + 3] * b[i + 3]) +
           (a[i + 4] * b[i + 4]) + (a[i + 5] * b[i + 5]) + (a[i + 6] * b[i + 6]) + (a[i + 7] * b[i + 7]);
  }

  for (int i = N - (N % 8); i < N; i++) {
    sum += a[i] * b[i];
  }
}

int main() {
  cout << "An integer is " << sizeof(int) << " bytes " << endl;

  int* a = new int[N];
  int* b = new int[N];

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }

  int sum = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  sum_normal(a, b, sum);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << "sum_normal: " << sum << " " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";

  delete[] a;
  delete[] b;

  return 0;
}