//Commit b7857af
 
namespace quda 	namespace quda {


  constexpr int size = 4096;
  static __constant__ char mobius_d[size]; // constant buffer used for Mobius coefficients for GPU kernel
  static void *mobius_h;                   // constant buffer used for Mobius coefficients for CPU kernel

  /**
     @brief Helper function for grabbing the constant struct, whether
     we are on the GPU or CPU.
  */
  template <typename real>
  inline __device__ __host__ complex<real> a_5(int s) {
#ifdef __CUDA_ARCH__
    return reinterpret_cast<const complex<real>*>(mobius_d)[s];
#else
    return reinterpret_cast<const complex<real>*>(mobius_h)[s];
#endif
  }

  template <typename Float, int nColor, QudaReconstructType reconstruct_>	  template <typename Float, int nColor, QudaReconstructType reconstruct_>
  struct DomainWall4DArg : WilsonArg<Float,nColor,reconstruct_> {	  struct DomainWall4DArg : WilsonArg<Float,nColor,reconstruct_> {
    typedef typename mapper<Float>::type real;	    typedef typename mapper<Float>::type real;
@@ -18,6 +35,7 @@ namespace quda {
    {	    {
      if (b_5 == nullptr || c_5 == nullptr) for (int s=0; s<Ls; s++) a_5[s] = a; // 4-d Shamir	      if (b_5 == nullptr || c_5 == nullptr) for (int s=0; s<Ls; s++) a_5[s] = a; // 4-d Shamir
      else for (int s=0; s<Ls; s++) a_5[s] = 0.5 * a / (b_5[s]*(m_5+4.0) + 1.0); // 4-d Mobius	      else for (int s=0; s<Ls; s++) a_5[s] = 0.5 * a / (b_5[s]*(m_5+4.0) + 1.0); // 4-d Mobius
      mobius_h = a_5;
    }	    }
  };	  };


@@ -39,10 +57,10 @@ namespace quda {
    int xs = x_cb + s*arg.dc.volume_4d_cb;	    int xs = x_cb + s*arg.dc.volume_4d_cb;
    if (xpay && kernel_type == INTERIOR_KERNEL) {	    if (xpay && kernel_type == INTERIOR_KERNEL) {
      Vector x = arg.x(xs, my_spinor_parity);	      Vector x = arg.x(xs, my_spinor_parity);
      out = x + arg.a_5[s] * out;	      out = x + a_5<real>(s) * out;
    } else if (kernel_type != INTERIOR_KERNEL && active) {	    } else if (kernel_type != INTERIOR_KERNEL && active) {
      Vector x = arg.out(xs, my_spinor_parity);	      Vector x = arg.out(xs, my_spinor_parity);
      out = x + (xpay ? arg.a_5[s] * out : out);	      out = x + (xpay ? a_5<real>(s) * out : out);
    }	    }


    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;	    if (kernel_type != EXTERIOR_KERNEL_ALL || active) arg.out(xs, my_spinor_parity) = out;
  }	  }
  // CPU Kernel for applying 4-d Wilson operator to a 5-d vector (replicated along fifth dimension)	  // CPU Kernel for applying 4-d Wilson operator to a 5-d vector (replicated along fifth dimension)
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>	  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  void domainWall4DCPU(Arg &arg)	  void domainWall4DCPU(Arg &arg)
  {	  {
    for (int parity= 0; parity < nParity; parity++) {	    for (int parity= 0; parity < nParity; parity++) {
      // for full fields then set parity from loop else use arg setting	      // for full fields then set parity from loop else use arg setting
      parity = nParity == 2 ? parity : arg.parity;	      parity = nParity == 2 ? parity : arg.parity;
      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume	      for (int x_cb = 0; x_cb < arg.threads; x_cb++) { // 4-d volume
        for (int s=0; s<arg.Ls; s++) {	        for (int s=0; s<arg.Ls; s++) {
          domainWall4D<Float,nDim,nColor,nParity,dagger,xpay,kernel_type>(arg, x_cb, s, parity);	          domainWall4D<Float,nDim,nColor,nParity,dagger,xpay,kernel_type>(arg, x_cb, s, parity);
        }	        }
      } // 4-d volumeCB	      } // 4-d volumeCB
    } // parity	    } // parity
  }	  }
  // GPU Kernel for applying 4-d Wilson operator to a 5-d vector (replicated along fifth dimension)	  // GPU Kernel for applying 4-d Wilson operator to a 5-d vector (replicated along fifth dimension)
  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>	  template <typename Float, int nDim, int nColor, int nParity, bool dagger, bool xpay, KernelType kernel_type, typename Arg>
  13 changes: 13 additions & 0 deletions13  
lib/dslash4_domain_wall_4d.cu
@@ -49,7 +49,20 @@ namespace quda {
    void apply(const cudaStream_t &stream) {	    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());	      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      Dslash<Float>::setParam(arg);	      Dslash<Float>::setParam(arg);
      typedef typename mapper<Float>::type real;
#ifdef JITIFY
      // we need to break the dslash launch abstraction here to get a handle on the constant memory pointer in the kernel module
      using namespace jitify::reflection;
      const auto kernel = DomainWall4DLaunch<void,0,0,0,false,false,INTERIOR_KERNEL,Arg>::kernel;
      auto instance = Dslash<Float>::program_->kernel(kernel)
        .instantiate(Type<Float>(),nDim,nColor,arg.nParity,arg.dagger,arg.xpay,arg.kernel_type,Type<Arg>());
      cuMemcpyHtoDAsync(instance.get_constant_ptr("quda::mobius_d"), mobius_h, QUDA_MAX_DWF_LS*sizeof(complex<real>), stream);
      Tunable::jitify_error = instance.configure(tp.grid,tp.block,tp.shared_bytes,stream).launch(arg);
#else
      cudaMemcpyToSymbolAsync(mobius_d, mobius_h, QUDA_MAX_DWF_LS*sizeof(complex<real>), 0,
                              cudaMemcpyHostToDevice, streams[Nstream-1]);
      Dslash<Float>::template instantiate<DomainWall4DLaunch,nDim,nColor>(tp, arg, stream);	      Dslash<Float>::template instantiate<DomainWall4DLaunch,nDim,nColor>(tp, arg, stream);
#endif
    }	    }


    TuneKey tuneKey() const { return TuneKey(in.VolString(), typeid(*this).name(), Dslash<Float>::aux[arg.kernel_type]); }