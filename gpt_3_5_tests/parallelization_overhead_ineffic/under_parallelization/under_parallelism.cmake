
//commit 20dae8a

if(GMX_OPENMP)
    # We should do OpenMP detection if we get here
    # OpenMP check must come before other CFLAGS!
    find_package(OpenMP)
    if(OPENMP_FOUND)
        # CMake on Windows doesn't support linker flags passed to target_link_libraries
        # (i.e. it treats /openmp as \openmp library file). Also, no OpenMP linker flags are needed.
        if(NOT (WIN32 AND NOT MINGW))
            if(CMAKE_COMPILER_IS_GNUCC AND GMX_PREFER_STATIC_OPENMP AND NOT APPLE)
                set(OpenMP_LINKER_FLAGS "-Wl,-static -lgomp -lrt -Wl,-Bdynamic -lpthread")
                set(OpenMP_SHARED_LINKER_FLAGS "")
            else()
                # Only set a linker flag if the user didn't set them manually
                if(NOT DEFINED OpenMP_LINKER_FLAGS)
                    set(OpenMP_LINKER_FLAGS "${OpenMP_C_FLAGS}")
                endif()
                if(NOT DEFINED OpenMP_SHARED_LINKER_FLAGS)
                    set(OpenMP_SHARED_LINKER_FLAGS "${OpenMP_C_FLAGS}")
                endif()
            endif()
        endif()
        if(MINGW)
            #GCC Bug 48659
            set(OpenMP_C_FLAGS "${OpenMP_C_FLAGS} -mstackrealign")
        endif()
    else()
        message(WARNING
                "The compiler you are using does not support OpenMP parallelism. This might hurt your performance a lot, in particular with GPUs. Try using a more recent version, or a different compiler. For now, we are proceeding by turning off OpenMP.")
        set(GMX_OPENMP OFF CACHE STRING "Whether GROMACS will use OpenMP parallelism." FORCE)
    endif()
endif()
gmx_dependent_cache_variable(GMX_OPENMP_MAX_THREADS
    "Maximum number of OpenMP Threads supported. Has to be 32 or a multiple of 64."
    STRING 32 GMX_OPENMP)
mark_as_advanced(GMX_OPENMP_MAX_THREADS)
math(EXPR MAX_THREAD_MOD "${GMX_OPENMP_MAX_THREADS} % 64")
if (NOT GMX_OPENMP_MAX_THREADS EQUAL 32 AND NOT ${MAX_THREAD_MOD} EQUAL 0)
    message(FATAL_ERROR "Only 32 or multiples of 64 supported for GMX_OPENMP_MAX_THREADS.")
endif()