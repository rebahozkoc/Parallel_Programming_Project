#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace std;
#define N 256*256*256

int main() {
  cout << "An integer is " << sizeof(int) << " bytes " << endl;

  vector<int> a(N);
  vector<int> b(N);

  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }

  int sum=0;
  auto t1 = std::chrono::high_resolution_clock::now();
  parallel_for(0, N, [&](int i) {
    sum += a[i] * b[i];
  });
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << "std::parallel: " << sum << " " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";
  

  return 0;
}