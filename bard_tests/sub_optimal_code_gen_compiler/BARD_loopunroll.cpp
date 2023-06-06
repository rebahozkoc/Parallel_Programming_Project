#include <iostream>
#include <chrono>
#include <vector>

using namespace std;
#define N 256*256*256

void sum_normal(vector<int> a, vector<int> b, int &sum) {
  for (int i = 0; i < N; i++) {
    sum += a[i] * b[i];
  }
}

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
  sum_normal(a, b, sum);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout << "sum_normal: " << sum << " " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";
  

  return 0;
}
