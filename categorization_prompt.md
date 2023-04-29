## PROMPT1: 

You are a high-performance computing engineer. I want you to classify the bugs I give to you that reduce the performance according to the categorization of this article.

An Empirical Study of High Performance
Abstract—Performance efficiency and scalability are the major design goals for high performance computing (HPC) applications. However, it is challenging to achieve high efficiency and scalability for such applications due to complex underlying hardware architecture, inefficient algorithm implementation, suboptimal code generation by the compilers, inefficient parallelization, and so on. As a result, the HPC community spends a significant effort detecting and fixing the performance bugs frequently appearing in scientific applications. However, it is important to accumulate the experience to guide the scientific software engineering community to write performance-efficient code. In this paper, we investigate open-source HPC applications to categorize the performance bugs and their fixes and measure the programmer’s effort and experience to fix them. For this purpose, we first perform a large-scale empirical analysis on 1729 HPC performance commits collected from 23 real-world projects. Through our manual analysis, we identify 186 performance issues from these projects. Furthermore, we study the root cause of these performance issues and generate a performance bug taxonomy for HPC applications. Our analysis identifies that inefficient algorithm implementation (39.3%), inefficient code for target micro-architecture (31.2%), and missing parallelism and inefficient parallelization (14.5%) are the top three most prevalent categories of performance issues for HPC applications. Additionally, to understand how the performance bugs are fixed, we analyze the performance fix commits and categorize them into eight performance fix types. We further measure the time it takes to discover a performance bug and the developer’s efforts and expertise required to fix them. The analysis identified that it’s difficult to localize performance inefficiencies, and once localized, fixes are complicated with a median patch size (LOC) of 35 lines and are mostly fixed by experienced developers. Index Terms—Empirical Study, HPC, Performance Optimization
…
RQ1: CAUSES OF INEFFICIENCY
We studied 186 performance commits from 23 open-source HPC projects and categorized them into 10 major categories. Figure 2 shows the detailed taxonomy of the HPC performance inefficiencies. In the rest of the section, we discuss each category of inefficiency and its root causes. 
A. Inefficient algorithm, data structure, computational kernel, and their implementation (IAD) We found that 73 out of 186 commits (39.3%) fix the performance issues that originated from the poor algorithm and data structure design and implementation. These inefficiencies include using computationally expensive operations (29), redundant operations (16), unnecessary operations (13), use of inefficient data structure (9), repeated function calls (3), and use of improper data types (3). 1) Computationally expensive operation: Applications with this inefficiency pattern perform a set of operations that incurs high-computational overhead at the runtime. Some sources of these expensive operations include using an inefficient algorithm or computational kernel, expensive runtime evaluation instead of compile-time evaluation, expensive data-structure traversals, and higher-precision arithmetic operations. However, from the commits, we observe that this computational overhead can be bypassed on many occasions using techniques such as compile-time evaluation, algorithmic strength reduction or approximation, caching, and reduced precision arithmetic. 

1 void bilateralKernel( 
2 ... 
3 -gauss2d[ly*window_size+lx] = exp( ((x*x) + (y*y)) 
4 - /(-2.f * variance_space)); 
5 +gauss2d[ly*window_size+lx] = __expf(((x*x) + (y*y)) 
6 + / variance_space); 

Listing 1. CUDA’s exp() function is computationally expensive than __expf().

For instance, the Listing 1 depicts the partial commit, ArrayFire-d0d87ab from ArrayFire. According to the commit, ArrayFire’s bilateral filter kernel uses CUDA’s [45] exponential function exp(). The exp() implements a double precision exponential function which is computationally expensive. On the other hand, CUDA implements a less computationally expensive alternative, __expf(), with fewer native instructions. However, __expf() suffers from accuracy loss compared to exp(). Since the algorithm can tolerate the loss of accuracy, using computationally expensive exp() instead of __expf() is causing performance inefficiency. 2) Redundant operations: This category of inefficiencies arises due to the iterative execution of an operation while the operation itself is invariant of the iteration, making iterative execution redundant. MFEM-2c9ee23 commit of MFEM reports such inefficiency. Listing 2 depicts the relevant changes of the commit. According to the commit, the MFEM’s MultTranspose function performs a multiplication of a dense matrix loc_prol with vectors and transposed in a hot loop via the MultTranspose function call. However, matrix loc_prol does not change over the iteration. As a result, based on the multiplication property of transpose, the transpose operation on each iteration is redundant and can be carried out outside the loop.

1 void OrderTransferOperator::MultTranspose(const ,→ Vector &x, Vector &y) const 
2 ... 
3 -subY.SetSize(loc_prol.Width()); 
4 +loc_prol.Transpose(); 
5 +subY.SetSize(loc_prol.Height()); 
6 ... 
7 for (int vd = 0; vd < vdim; vd++){ 
8 ... 
9 - loc_prol.MultTranspose(subX, subY); 
10 + loc_prol.Mult(subX, subY); 

Listing 2. Redundant transpose operation in loop

3) Unnecessary operations: In such cases, the inefficient code performs a computation or data access whose results were never used in the algorithm. For instance, OpenFOAM-dev-91e84b9 commit identifies such inefficiency. The Listing 3 depicts the partial commit where OpenFOAMdev’s stochasticCollision algorithm checks whether all parcels collide regardless of their associativity to cells. However, since parcels belonging to different cells will never collide, many collision checks are unnecessary, resulting in poor performance of the algorithm.

1 void Foam::ORourkeCollision<CloudType>::collide(
,→ const scalar dt)
2 -forAllIter(typename CloudType,this->owner(),iter1){
3 - forAllIter(typename CloudType,this->owner(),
,→ iter2)
4 +// Create a list of parcels in each cell
5 +List<DynamicList<parcelType*>> pInCell(this->owner
,→ ().mesh().nCells());
6 + // iterate the cell list
7 + for (label celli=0;celli<this->owner().mesh().
,→ nCells(); celli++){
8 + // compare parcels within the cell
9 + forAll(pInCell[celli], i){
10 + for (label j=i+1; j<pInCell[celli].size
,→ (); j++){
11 bool massChanged = collideParcels(dt, p1,
,→ p2, m1, m2);
Listing 3. Unnecessary traversal in collision detection

## PROMPT 2:

4) Inefficient data-structure: We found that 7 out 9 commits of this category fix performance bugs originate from choosing an inefficient data structure library. For instance, TileDB-d51b082 commit reported that the read query implemented using forward_list from C++’s Standard Template Library (STL) [46] is performing slower. The forward_list implements a linked list data structure. As a result, traversing forward_list results in a poor data locality compared to other cache-efficient data structures such as STL’s vector. B. Inefficient code for underlying micro-architecture (MA) We found 58 commits (31.8%) to fix the performance bugs that originated from inefficient code for underlying hardware micro-architecture. High-performance computing application’s performance is susceptible to memory access latency, hazards in the instruction pipeline, branch divergence, and many other micro-architectural nuances. 1) Inefficiency due to memory/data locality: The principle of locality heavily influences modern processors design choices. According to the principle, applications tend to access a small set of data repeatedly. Modern processors implement hierarchical data storage to reduce memory latency of these frequently accessed data. Register memory, the fastest data storage, resides next to the processor. The processors implement a limited number of registers. As a result, efficient utilization of the registers is important for high performance. Cache memories residing next to registers can contain more data at a relatively higher access latency than the registers. Again, processors implement multi-level cache, where data access from the far-caches incurs higher latency. Modern GPU architectures further implement various types of cachememory targeting application’s memory access patterns. Due to the complexity of the memory hierarchy of modern processors, it is often challenging to write efficient code. Our empirical study finds 36 commits (19.4%) that fix these various memory/data locality challenges. a) Inefficient cache access: We found 18 commits fix inefficiencies arise due to poor cache utilization resulting in high memory access latency. One such example is false sharing. False sharing occurs when two or more threads run on two or more CPU cores and access two different memory addresses in the same cache line. Such access causes cache line eviction from one CPU core and vice versa and increases cache miss. For example, the Kokkos75fd8bc commit reported a false sharing in it’s random number pools which are shared by multiple threads. Since the array of elements per thread is small, they share cache line and causes false-sharing. The commit fixes the issue by padding the array so that elements in the array reside in the separate cache line. The commit reported 200x improvement for 20 threads on the Intel Skylake machine running the random number test case. Another example of such inefficiency is non-linear data access that causes poor spatial locality. Spatial locality, a kind of data locality, states that given data access, nearby memory locations will be referenced soon. To realize the property, a cache-line, the basic building block of a cache, holds consecutive memory addresses. Any high-performance application traverses memory non-linearly will fail to utilize the cache-line locality and incur significant memory latency. For instance, the cp2k-7b34ac6 commit identifies that cp2k‘s Gamma_P_ia had poor spatial locality due to its access pattern. b) Inefficient GPU-memory access: Modern GPU processors implement multiple device memory spaces such as global, local, shared, constant, texture memory, and register files [7]. Depending on the data-access pattern, each type of memory has advantages and disadvantages. From our empirical study, we observe that 18 commits (9.7%) fix the inefficient use of GPU memory. We found 6 GPU memory-related commits using highlatency global memory. All threads can access data stored in global memory in the GPU. However, global memory has more access latency than shared, constant, and textured memory. As a result, it is important to reduce global memory use when possible. For example, the GROMACS-8aa14d1 commit identifies that GROMACS’ bonded kernel performs a reduction operation across all the threads using atomic instruction on global memory. However, performing the reduction operation on high-latency global memory is unnecessary in many cases. For example, the threads residing in the same warp can perform the reduction operations in low-latency shared memory, avoiding expensive global memory access. However, the implementation fails to realize this property and suffers from a performance bottleneck. Due to limited space in register files, efficient register use is important to reduce register spilling. We found three commits that fix GPU-register spilling. For instance, the QUDAb7857af commit identifies register spilling in Lattice-QUDA’s dslash4_domain_Wall_4d kernel. The kernel used registers to store coefficients. However, since the coefficients are not updated during kernel execution, keeping them in the registers instead of constant memory increases register pressure, resulting in register spilling. We found two commits that fix an inefficient host-device communication. Data movement between the host CPU and the GPU causes significant performance overhead. The CUDA cudaMemcpy copies data between the host and device synchronously. Due to this blocking call, it fails to overlap the computation and communication and thus fails to mask communication latency. For instance, the ginkgo-154aafb commit identifies that ginkgo’s residual_norm_reduction performs a synchronous data copy using cudaMemcpy. However, these high latency blocking function calls can be avoided by copying the data asynchronously during kernel launch time. If the source or destination in host memory is not allocated as pinned, the host-device memory copy will require to transfer the data to a pinned memory at first. This causes an extra data copy and causes significant performance overhead. One such example is identified in the OpenMM-926e7b9 commit, where OpenMM’s CudaCalcAmoebaVdwForceKernel transfers the context parameter VdwLambda from host to device using unpinned host memory. 2) Sub-optimal code generation by compiler: On several occasions, the compiler fails to statically reason about the most opportunities of a source code. We found 9 commits (4.8%) fixed the sub-optimal code generated by the compiler. a) Loop unrolling: Loop unrolling is a type of loop optimization where the loop body is replicated multiple times to reduce the loop iterations. Loop unrolling enables opportunities for instruction-level parallelism where the processor can perform simultaneous execution of independent instructions. Compilers often miss the opportunity for loop unrolling due to the inability to statically determine the number of loop iterations and the memory access patterns. We found 5 commits fix missed opportunities for loop unrolling. For instance, the ArrayFire-7f3fe1e commit identifies the missed opportunity of loop unrolling, resulting in suboptimal code execution. As shown in Listing 4, ArrayFire’s transpose kernel uses #pragma unroll directive to guide the CUDA compiler to unroll the loop. However, the compiler fails to determine the memory accesses statically during compile time.


1 void transpose(Param out, CParam in, 
2 dim_type nonBatchBlkSize){ 
3 ... 
4 #pragma unroll 
5 - for (dim_type repeat = 0; repeat < TILE_DIM; 
6 - repeat += blockDim.y) { 
7 + for (int repeat = 0; repeat < TILE_DIM; 
8 + repeat += THREADS_Y) { 

Listing 4. Commit identifies missed loop unroll opportunity
b) Function inlining: Function inlining is a compiler optimization technique where the body of the called function replaces a function call. On the one hand, function inlining can eliminate function call-related overhead, such as passing arguments and handling return value, and reduces register spilling. On the other hand, this optimization further opens many other intra-procedural optimization opportunities. However, the performance impact of function inlining is not always obvious and depends on the function, and it’s invocation pattern. Previous literature has shown that function inlining optimization may or may not occur depending on the compiler version [9]. In our empirical study, we have found 3 performance commits manually direct the compiler to inline a function. For instance, the libMesh-e0374af commit identifies that libMesh’s BoundingBox::contains_point method implemented small helper function is_between to make contains_point() more readable. However, the compiler does not generate is_between as an inline and misses many optimization opportunities.

1 +#pragma unroll
2 for (int i=0; i<M; i++){
3 - for (int j=0; j<N; j++){
4 - ... 
5 - copy(v[(chirality*M+i)*N+j], clover[parity] 
6 - [(padIdx*stride + x)*N + intIdx%N]); 
7 - ... 
8 - } 
9 + Vector vecTmp = vector_load(clover 
10 + + parity*offset, x +stride*(chirality*M+i)); 
11 + for (int j=0; j<N; j++)
12 + copy(v[(chirality*M+i)*N+j], 
13 + reinterpret_cast(&vecTmp)[j]); 
14 } 

Listing 5. Vectorization of the loop achieves 1.5x speedup 

C. Missing parallelism (MP) We found that around 13 commits (7%) fix the missing parallelism in the original code. We observe a range of parallelism is introduced, including Vector parallelism/SIMD Parallelism (5), GPU parallelism (2), Instruction level parallelism (1), and Task parallelism (5). 1) SIMD parallelism:: The QUDA-5f028db commit identifies the missing SIMD parallelism. The listing 5 depicts the partial commit. As shown in the listing, QUDA‘s FloatNOrder::load() function has an inner loop performing consecutive memory access. However, the code fails to use SIMD intrinsic to get the benefits of parallel load/store operation. 2) GPU parallelism:: The commit HYPRE-a05d194 identifies missing GPU parallelism. The Listing 6 highlights the commit where it shows that HYPRE’s hypre_ParCSRRelax function iterates a loop sequentially to read and write to an array and fails to use GPU parallelism

1 +#if defined(HYPRE_USING_OPENMP_OFFLOAD) 
2 + int num_teams = (num_rows+num_rows%1024)/1024; 
3 + #pragma omp target teams distribute 
4 + parallel for private(i) num_teams(num_teams) 
5 + thread_limit(1024) is_device_ptr(u_data,v_data, 
6 + l1_norms) 
7 +#endif 
8 for (i = 0; i < num_rows; i++)

Listing 6. Missed GPU parallelism in HYPRE
3) Task parallelism: The commit GOMC37a6bfd identifies that in GOMC’s CalculateEnergy::ParticleInter function, the energy calculation of Monte Carlo simulation could not take full advantage of the parallelism provided by multi-core processors. As shown in Listing 7, the outer loop first iterates over all the trials, then performs the energy calculation, which could be divided into independent tasks for a partition range and executed in different threads to exploit task parallelism for efficiency

1 - for (t = 0; t < trials; ++t){ 
2 +#pragma omp parallel sections private(start, end)
3 + { 
4 + #pragma omp section 
5 + if(Schedule(start, end, p, 0, trials)) 
6 + …

Listing 7. Missing task parallelism in GOMC. 

D. Inefficient parallelization (PO) We found that 13 commits (7%) fix inefficient parallel code regions. Among these commits, 11 commits fix the inefficient work partitioning for the target accelerator. Efficiently implementing a parallel algorithm on GPU architecture requires consideration of underlying GPU architecture. Modern GPU architectures are complex consisting of several streaming multiprocessors (SM) where each SM executes a set of threads known as warps. To fully utilize the GPU parallel capability, developers require to correctly split the parallel workload. An inefficient workload mapping results in load imbalance and high thread management overhead. The commit QUDA-4ef8d8a identifies such inefficiency in QUDA’s Multi-Reduce kernel. In the kernel, the block size, a CUDA programming abstraction, was incorrectly set that causing under-utilization of warps in SM. E. Inefficient Concurrency control (ICS) Our study found 7 commits (3.8%) fix the concurrency and synchronization-related performance bugs. These performance bugs include unnecessary locks (3), inappropriate lock granularity (1), use of inefficient locking mechanism (1), and unnecessary thread synchronization (2). The commit OpenBLAS-3119b2a identifies unnecessary locks in OpenBLAS’s alloc_mmap function. Listing 8 depicts the case, where the function holds the alloc_lock while accessing global data structure release_info. However, the code causes performance inefficiency when OpenMP thread parallelism is enabled. Since OpenMP already imposes a lock, this explicit locking is unnecessary. 

1 static void *alloc_mmap(void *address){ 
2 +#if defined(SMP) && defined(USE_OPENMP) 
3 + LOCK_COMMAND(&alloc_lock); 
4 +#endif 
5 release_info[release_pos].address = map_address; 
6 release_info[release_pos].func = ,→ alloc_mmap_free; 
7 release_pos ++; 
8 +#if defined(SMP) && defined(USE_OPENMP) 
9 + UNLOCK_COMMAND(&alloc_lock); 
10 +#endif 

Listing 8. Unnecessary locking for OpenMP based multi-threading.

## PROMPT 3:

F. Inefficient memory management (IMM) We found 13 commits (7%) that fix inefficient memory management-related issues, including memory leak, redundant memory allocation, and repeated calls for allocation. The OpenBlas-d744c95 commit identifies repeated allocation in OpenBLAS’s exec_threads function. The function allocates buffer by calling blas_memory_alloc function. The exec_threads function is called by all the participating threads, which incurs a runtime overhead.

G. Other forms of inefficiencies Our study finds a small set of other forms of inefficiencies including unintentional programming logic error (PE), IO inefficiency (IO), compiler regression (CR), and unnecessary process communication (UPC). 

Here is the detailed form of categories in JSON format:

        {
            "HPC performance bugs": [
            {
                "Inefficient coding for target micro-architecure": [
                [
                    {
                    "Memory/Data locality": [
                        {
                        "Intra-thread data locality": [
                            "cache locality",
                            "Memory alignment",
                            "GPU Register spilling",
                            "Host-GPU device data communication",
                            "GPU memory"
                        ]
                        },
                        "Inter-thread Data locality"
                    ]
                    }
                ],
                [
                    {
                    "Micro-architectural inefficiency": [
                        "Missed compiler loop optimization",
                        "Loop carried dependence",
                        "suboptimal computational kernel size for the target",
                        "Use of slow instruction set",
                        "suboptimal API for the target architecture"
                    ]
                    }
                ]
                ]
            },
            {
                "Missing parallelism": [
                "Vector/SIMD parallelism",
                "GPU parallelism",
                "Instruction level parallelism",
                "Task parallelism"
                ]
            },
            {
                "Parallelization overhead/inefficiency": [
                "small parallel region",
                "Inefficeint thread mapping / inefficient block size / Load imbalance",
                "Under-parallelization",
                "Over-Parallelization"
                ]
            },
            {
                "Inefficient Concurrency control and synchronization": [
                "Unncessary locks",
                "Unncessary strong memory consistency",
                "Lock management overhead",
                "Unnecessary synchronization"
                ]
            },
            "Unnecessary process communiction",
            {
                "Inefficient algorithm /data-structure and their implementation": [
                {
                    "Unnecessary operation/traversal/function call": [
                    "Unnecessary computation",
                    "Unnecessary function call",
                    "Inefficient sparse array computation",
                    "Unnecessary traversal",
                    "Unnecessary itermediate variable usage"
                    ]
                },
                {
                    "Redundant operation": [
                    "Redundant traversal",
                    "Redundant allocation",
                    "Redundant computation",
                    "Unnecessary branching"
                    ]
                },
                {
                    "Expensive operation": [
                    "Expensive algorithm",
                    "Expensive runtime evaluation",
                    "Expensive many memory references",
                    "Expensive atomic call",
                    "Expensive computational kernel for a target architecture",
                    "expensive high precision calculation",
                    "Expensive exponential operation(O(n) overhead)",
                    "Expensive tensor calculation",
                    "Expensive type-casting",
                    "Expensive library function call",
                    "Expensive integer division",
                    "Expensive function call",
                    "Expensive kernel operation for small matrix"
                    ]
                },
                "Frequent  function call",
                {
                    "Inefficient data-structure library": [
                    "Slower data structure",
                    "Inefficient library API call"
                    ]
                },
                {
                    "Usage of improper data type": "Slowdown due to wrong data type"
                }
                ]
            },
            {
                "Inefficient memory management": [
                "memory leak",
                "repreated memory allocation",
                "Redundant memory allocation",
                "Slower memory allocation library call",
                "Insufficient memory",
                "unnecessary data copy"
                ]
            },
            "I/O inefficiency",
            "Unintentional Programming logic error",
            "Inefficiency due to new compiler version"
            ]
        }


