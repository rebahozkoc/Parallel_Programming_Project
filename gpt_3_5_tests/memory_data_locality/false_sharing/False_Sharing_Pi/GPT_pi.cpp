#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

long long num_steps = 1024 * 1024 * 1024;
double step;

// Add padding to avoid false sharing
#define CACHE_LINE_SIZE 64
#define PAD_SIZE (CACHE_LINE_SIZE/sizeof(double))
struct padded_double {
    double value;
    double padding[PAD_SIZE];
};

int main (int argc, char** argv)
{
  long long int i;
  double x, pi;
  padded_double sum;
  sum.value = 0.0;
  double start_time, run_time;

  start_time = omp_get_wtime();

  //code 
  step = 1.0/(double) num_steps;

#pragma omp parallel for 
  for (int i = 0; i < num_steps; i++) {
    double x = (i + 0.5)*step;
    sum.value += 4.0/(1.0+x*x);
  }
  pi = step * sum.value;

  run_time = omp_get_wtime() - start_time;
  printf(" pi                is 3.14159265358979323846264338327950");
  printf("\n pi with %lld steps is %.20lf in %lf seconds\n ",num_steps,pi,run_time);
}	  
