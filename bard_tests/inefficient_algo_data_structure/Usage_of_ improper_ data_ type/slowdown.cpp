// 0caf143
int sgemm_kernel_direct_performant(BLASLONG M, BLASLONG N, BLASLONG K)
{
	int mnk = M * N * K;
	/* large matrixes -> not performant */
	if (mnk >= 28 * 512 * 512)
		return 0;