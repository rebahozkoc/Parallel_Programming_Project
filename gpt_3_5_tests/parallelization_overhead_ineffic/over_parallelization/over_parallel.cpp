// Commit 0c1c903
int exec_blas(BLASLONG num, blas_queue_t *queue){
  BLASLONG i, buf_index;
  if ((num <= 0) || (queue == NULL)) return 0;
#ifdef CONSISTENT_FPCSR
  for (i = 0; i < num; i ++) {
    __asm__ __volatile__ ("fnstcw %0"  : "=m" (queue[i].x87_mode));
    __asm__ __volatile__ ("stmxcsr %0" : "=m" (queue[i].sse_mode));
  }
#endif
  while(true) {
    for(i=0; i < MAX_PARALLEL_NUMBER; i++) {
#ifdef HAVE_C11
      _Bool inuse = false;
      if(atomic_compare_exchange_weak(&blas_buffer_inuse[i], &inuse, true)) {
#else
      if(blas_buffer_inuse[i] == false) {
        blas_buffer_inuse[i] = true;
#endif
        buf_index = i;
        break;
      }
    }
    if(i != MAX_PARALLEL_NUMBER)
      break;
  }

#pragma omp parallel for schedule(OMP_SCHED)
  for (i = 0; i < num; i ++) {

#ifndef USE_SIMPLE_THREADED_LEVEL3
    queue[i].position = i;
#endif
    exec_threads(&queue[i], buf_index);
  }
#ifdef HAVE_C11
  atomic_store(&blas_buffer_inuse[buf_index], false);
#else
  blas_buffer_inuse[buf_index] = false;
#endif
  return 0;
}
#endif