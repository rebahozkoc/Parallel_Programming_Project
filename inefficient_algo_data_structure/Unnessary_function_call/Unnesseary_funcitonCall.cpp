void cudaColorSpinorField::destroyTexObject() {
    if (isNative() && texInit) {
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(tex);
      if (ghost_bytes) {
	cudaDestroyTextureObject(ghostTex[0]);
Expand All
	@@ -448,7 +447,6 @@ namespace quda {

  void cudaColorSpinorField::destroyGhostTexObject() {
    if (isNative() && ghostTexInit) {
      cudaDeviceSynchronize();
      cudaDestroyTextureObject(ghostTex[0]);
      cudaDestroyTextureObject(ghostTex[1]);
      if (precision == QUDA_HALF_PRECISION) {
Expand Down
Expand Up
	@@ -979,7 +977,6 @@ namespace quda {
    if (!initComms || nFaceComms != nFace || bufferMessageHandler != bufferPinnedResizeCount) {

      // if we are requesting a new number of faces destroy and start over
      destroyIPCComms();
      destroyComms();

      if (siteSubset != QUDA_PARITY_SITE_SUBSET) 
Expand Down
Expand Up
	@@ -1237,6 +1234,7 @@ namespace quda {
      checkCudaError();
    }

    createIPCComms();
  }