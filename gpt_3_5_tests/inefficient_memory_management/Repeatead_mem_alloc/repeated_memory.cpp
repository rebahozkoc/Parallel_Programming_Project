  csParam.create = QUDA_ZERO_FIELD_CREATE;
    cudaColorSpinorField y(b, csParam);

    // compute initial residual
    double r2 = 0.0;
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      // Compute r = b - A * x
      mat(r, x);
      r2 = blas::xmyNorm(b, r);
      if (b2 == 0) b2 = r2;
      // y contains the original guess.
Expand All
	@@ -151,12 +171,10 @@ namespace quda
    if (param.deflate && param.maxiter > 1) {
      // Deflate and accumulate to solution vector
      eig_solve->deflate(y, r, evecs, evals, true);
      mat(r, y);
      r2 = blas::xmyNorm(b, r);
    }

    csParam.setPrecision(param.precision_sloppy);
    cudaColorSpinorField tmpSloppy(x, csParam);
    cudaColorSpinorField Ap(x, csParam);

    cudaColorSpinorField *r_sloppy;
Expand All
	@@ -178,8 +196,8 @@ namespace quda
      x_sloppy = new cudaColorSpinorField(x, csParam);
    }

    cudaColorSpinorField &xSloppy = *x_sloppy;
    cudaColorSpinorField &rSloppy = *r_sloppy;

    blas::zero(x);
    if (&x != &xSloppy) blas::zero(xSloppy);
Expand Down
Expand Up
	@@ -243,7 +261,7 @@ namespace quda

    while (!convergence(r2, heavy_quark_res, stop, param.tol_hq) && k < param.maxiter) {

      matSloppy(Ap, *p, tmpSloppy);

      double sigma;
      pAp = reDotProduct(*p, Ap);
Expand Down
Expand Up
	@@ -292,15 +310,15 @@ namespace quda
        axpy(alpha, *p, xSloppy); // xSloppy += alpha*p
        xpy(xSloppy, y);          // y += x
        // Now compute r
        mat(r, y, x); // x is just a temporary here
        r2 = xmyNorm(b, r);

        if (param.deflate && sqrt(r2) < maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflate(y, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, y, x);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2);
Expand Down
Expand Up
	@@ -367,10 +385,10 @@ namespace quda

    if (k == param.maxiter) warningQuda("Exceeded maximum iterations %d", param.maxiter);

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("CG: Reliable updates = %d\n", rUpdate);

    // compute the true residual
    mat(r, x, y);
    double true_res = xmyNorm(b, r);
    param.true_res = sqrt(true_res / b2);

Expand All
	@@ -383,6 +401,12 @@ namespace quda
    profile.TPSTOP(QUDA_PROFILE_EPILOGUE);
    profile.TPSTART(QUDA_PROFILE_FREE);

    if (K) { // These are only needed if preconditioning is used
      delete minvrPre;
      delete rPre;