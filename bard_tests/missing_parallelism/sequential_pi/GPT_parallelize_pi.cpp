#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

long long num_steps = 1024 * 1024 * 1024;
double step;

int main (int argc, char** argv)
{
  double start_time, run_time;

  start_time = omp_get_wtime();

  double step;
  long long int i; double x, pi, sum = 0.0;
  step = 1.0/(double) num_steps;

  #pragma omp parallel for private(x) reduction(+:sum)
  for (i=0;i< num_steps; i++){
    x = (i+0.5)*step;
    sum = sum + 4.0/(1.0+x*x);
  }

  pi = step * sum;

  run_time = omp_get_wtime() - start_time;
  printf(" pi                is 3.14159265358979323846264338327950");
  printf("\n pi with %lld steps is %.20lf in %lf seconds\n ",num_steps,pi,run_time);
}