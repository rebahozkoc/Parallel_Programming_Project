#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

long long num_steps = 1024 * 1024 * 1024;
double step;

int main (int argc, char** argv)
{
  long long int i;
  double x, pi, sum = 0.0;
  double start_time, run_time;

  start_time = omp_get_wtime();

  //code 
  step = 1.0/(double) num_steps;
  double* partial_sum = new double[omp_get_max_threads()];

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    partial_sum[tid] = 0.0;

    #pragma omp for
    for (int i = 0; i < num_steps; i++) {
      double x = (i + 0.5)*step;
      partial_sum[tid] += 4.0/(1.0+x*x);
    }
  }

  for (int i = 0; i < omp_get_max_threads(); i++) {
    sum += partial_sum[i];
  }

  pi = step * sum;

  delete[] partial_sum;
  
  run_time = omp_get_wtime() - start_time;
  printf(" pi                is 3.14159265358979323846264338327950");
  printf("\n pi with %lld steps is %.20lf in %lf seconds\n ",num_steps,pi,run_time);
}
