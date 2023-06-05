//compile with: g++ prefetch.cpp  -std=c++11 -O3 
//use -fopt-info-vec-optimized for vectorization info verbose
//run: ./a.out

#include <iostream>
#include <chrono>
using namespace std;

#define N 1024 * 1024 * 1024

void sum_norm(int* a, int* b, int& sum) {
  sum = 0;
  for (int i = 0; i < N; i++){
    sum += a[i]*b[i];
  }

}
int main() {
  cout << "An integer is " << sizeof(int) << " bytes "<< endl;

  int* a = new int[N];
  int* b = new int[N];
  
  for(int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i;
  }
  cout << a << " " << b << endl;
  
  int sum;
  auto t1 = std::chrono::high_resolution_clock::now();
  sum_norm(a, b, sum);
  auto t2 = std::chrono::high_resolution_clock::now();
  cout <<"Without prefetching " <<sum << " ns " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << " milliseconds\n";
  return 0;
}
