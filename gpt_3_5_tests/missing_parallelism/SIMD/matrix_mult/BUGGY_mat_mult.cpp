//compile with : g++ BUGGY_mat_mult_avx.cpp  -std=c++11 -O3 -march=native
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
  float scratchpad[8];
  for (int i=0; i<row; i++) {
    for (int j=0; j<col; j++) {
      w[i][j]=(float)(rand()%1000)/800.0f;
    }
  }
  for (int j=0; j<col; j++) {
    x[j]=(float)(rand()%1000)/800.0f;
  }
 
  clock_t t1, t2;
 
  t1 = clock();
  for (int r = 0; r < num_trails; r++)
    for(int j = 0; j < row; j++)
      {
	float sum = 0;
	float *wj = w[j];
	for(int i = 0; i < col; i++) {
	 sum += wj[i] * x[i];
	}
	y[j] = sum;
  }
  t2 = clock();
  float diff = (((float)t2 - (float)t1) / CLOCKS_PER_SEC ) * 1000;
  cout<<"Without SIMD Time taken: "<<diff<<endl;
   cout<< "Vector result: \n ";
    cout << "\t";
 for (int i=0; i<row; i++) {
    cout<<y[i]<<", ";
  }
  cout<<endl;
  return 0;
}
