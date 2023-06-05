// Commit 0c1c903
#pragma omp parallel for schedule(OMP_SCHED)
  for (i = 0; i < num; i ++) {

#ifndef USE_SIMPLE_THREADED_LEVEL3