// Commit 0c1c903
static void exec_threads(blas_queue_t *queue, int buf_index){
  void *buffer, *sa, *sb;
  int pos=0, release_flag=0;
  buffer = NULL;
  sa = queue -> sa;
  sb = queue -> sb;
#ifdef CONSISTENT_FPCSR
  __asm__ __volatile__ ("ldmxcsr %0" : : "m" (queue -> sse_mode));
  __asm__ __volatile__ ("fldcw %0"   : : "m" (queue -> x87_mode));
#endif
  if ((sa == NULL) && (sb == NULL) && ((queue -> mode & BLAS_PTHREAD) == 0)) {
    pos = omp_get_thread_num();
    buffer = blas_thread_buffer[buf_index][pos];
    //fallback
    if(buffer==NULL) {
      buffer = blas_memory_alloc(2);
      release_flag=1;
    }
    if (sa == NULL) {
      sa = (void *)((BLASLONG)buffer + GEMM_OFFSET_A);
      queue->sa=sa;
    }
    if (sb == NULL) {
      if (!(queue -> mode & BLAS_COMPLEX)){
#ifdef EXPRECISION
	if (queue -> mode & BLAS_XDOUBLE){
	  sb = (void *)(((BLASLONG)sa + ((QGEMM_P * QGEMM_Q * sizeof(xdouble)
					  + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
	} else
#endif
	  if (queue -> mode & BLAS_DOUBLE){
	    sb = (void *)(((BLASLONG)sa + ((DGEMM_P * DGEMM_Q * sizeof(double)
					    + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
	  } else {
	    sb = (void *)(((BLASLONG)sa + ((SGEMM_P * SGEMM_Q * sizeof(float)
					    + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
	  }
      } else {
#ifdef EXPRECISION
	if (queue -> mode & BLAS_XDOUBLE){
	  sb = (void *)(((BLASLONG)sa + ((XGEMM_P * XGEMM_Q * 2 * sizeof(xdouble)
					  + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
	} else
#endif
	  if (queue -> mode & BLAS_DOUBLE){
	    sb = (void *)(((BLASLONG)sa + ((ZGEMM_P * ZGEMM_Q * 2 * sizeof(double)
					    + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
	  } else {
	    sb = (void *)(((BLASLONG)sa + ((CGEMM_P * CGEMM_Q * 2 * sizeof(float)
					    + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);
	  }
      }
      queue->sb=sb;
    }
  }
  if (queue -> mode & BLAS_LEGACY) {
    legacy_exec(queue -> routine, queue -> mode, queue -> args, sb);
  } else
    if (queue -> mode & BLAS_PTHREAD) {
      void (*pthreadcompat)(void *) = queue -> routine;
      (pthreadcompat)(queue -> args);
    } else {
      int (*routine)(blas_arg_t *, void *, void *, void *, void *, BLASLONG) = queue -> routine;
      (routine)(queue -> args, queue -> range_m, queue -> range_n, sa, sb, queue -> position);
    }
  if (release_flag) blas_memory_free(buffer);
}
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