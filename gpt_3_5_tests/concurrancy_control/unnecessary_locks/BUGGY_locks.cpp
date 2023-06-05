

  if (map_address != (void *)-1) {
    LOCK_COMMAND(&alloc_lock);
    release_info[release_pos].address = map_address;
    release_info[release_pos].func    = alloc_mmap_free;
    release_pos ++;
    UNLOCK_COMMAND(&alloc_lock);
  }

#ifdef OS_LINUX
Expand Down
Expand Up
	@@ -601,14 +605,18 @@ static void *alloc_mmap(void *address){
#if defined(OS_LINUX) && !defined(NO_WARMUP)
  }
#endif
  LOCK_COMMAND(&alloc_lock);

  if (map_address != (void *)-1) {
    release_info[release_pos].address = map_address;
    release_info[release_pos].func    = alloc_mmap_free;
    release_pos ++;
  }
  UNLOCK_COMMAND(&alloc_lock);

  return map_address;
}
Expand Down
Expand Up
	@@ -1007,7 +1015,10 @@ void *blas_memory_alloc(int procpos){
    NULL,
  };
  void *(**func)(void *address);
  LOCK_COMMAND(&alloc_lock);

  if (!memory_initialized) {

Expand Down
Expand Up
	@@ -1041,7 +1052,9 @@ void *blas_memory_alloc(int procpos){
    memory_initialized = 1;

  }
  UNLOCK_COMMAND(&alloc_lock);

#ifdef DEBUG
  printf("Alloc Start ...\n");
Expand All
	@@ -1056,12 +1069,15 @@ void *blas_memory_alloc(int procpos){

  do {
    if (!memory[position].used && (memory[position].pos == mypos)) {
      LOCK_COMMAND(&alloc_lock);
/*      blas_lock(&memory[position].lock);*/

      if (!memory[position].used) goto allocation;

      UNLOCK_COMMAND(&alloc_lock);
/*      blas_unlock(&memory[position].lock);*/
    }

Expand All
	@@ -1076,12 +1092,15 @@ void *blas_memory_alloc(int procpos){

  do {
/*    if (!memory[position].used) { */
      LOCK_COMMAND(&alloc_lock);
/*      blas_lock(&memory[position].lock);*/

      if (!memory[position].used) goto allocation;
      
      UNLOCK_COMMAND(&alloc_lock);
/*      blas_unlock(&memory[position].lock);*/
/*    } */

Expand All
	@@ -1098,8 +1117,10 @@ void *blas_memory_alloc(int procpos){
#endif

  memory[position].used = 1;

  UNLOCK_COMMAND(&alloc_lock);
/*  blas_unlock(&memory[position].lock);*/

  if (!memory[position].addr) {
Expand Down
Expand Up
	@@ -1146,9 +1167,13 @@ void *blas_memory_alloc(int procpos){

    } while ((BLASLONG)map_address == -1);

    LOCK_COMMAND(&alloc_lock);
    memory[position].addr = map_address;
    UNLOCK_COMMAND(&alloc_lock);

#ifdef DEBUG
    printf("  Mapping Succeeded. %p(%d)\n", (void *)memory[position].addr, position);
Expand All
	@@ -1165,7 +1190,9 @@ void *blas_memory_alloc(int procpos){

  if (memory_initialized == 1) {

    LOCK_COMMAND(&alloc_lock);

    if (memory_initialized == 1) {

Expand All
	@@ -1174,8 +1201,9 @@ void *blas_memory_alloc(int procpos){
      memory_initialized = 2;
    }

    UNLOCK_COMMAND(&alloc_lock);

  }
#endif

Expand All
	@@ -1202,8 +1230,9 @@ void blas_memory_free(void *free_area){
#endif

  position = 0;
  LOCK_COMMAND(&alloc_lock);

  while ((position < NUM_BUFFERS) && (memory[position].addr != free_area))
    position++;

Expand All
	@@ -1217,7 +1246,9 @@ void blas_memory_free(void *free_area){
  WMB;

  memory[position].used = 0;
  UNLOCK_COMMAND(&alloc_lock);

#ifdef DEBUG
  printf("Unmap Succeeded.\n\n");
Expand All
	@@ -1232,8 +1263,9 @@ void blas_memory_free(void *free_area){
  for (position = 0; position < NUM_BUFFERS; position++)
    printf("%4ld  %p : %d\n", position, memory[position].addr, memory[position].used);
#endif
  UNLOCK_COMMAND(&alloc_lock);

  return;
  