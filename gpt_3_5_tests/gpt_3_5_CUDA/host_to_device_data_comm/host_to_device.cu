//commit 154aafb


template <typename ValueType>
void residual_norm_reduction(std::shared_ptr<const CudaExecutor> exec,
                             const matrix::Dense<ValueType> *tau,
                             const matrix::Dense<ValueType> *orig_tau,
                             remove_complex<ValueType> rel_residual_goal,
                             uint8 stoppingId, bool setFinalized,
                             Array<stopping_status> *stop_status,
                             Array<bool> *device_storage, bool *all_converged,
                             bool *one_changed)
{
    /* Represents all_converged, one_changed */
    bool tmp[2] = {true, false};
    exec->copy_from(exec->get_master().get(), 2, tmp,
                    device_storage->get_data());

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(tau->get_size()[1], block_size.x), 1, 1);

    residual_norm_reduction_kernel<<<grid_size, block_size, 0, 0>>>(
        tau->get_size()[1], rel_residual_goal,
        as_cuda_type(tau->get_const_values()),
        as_cuda_type(orig_tau->get_const_values()), stoppingId, setFinalized,
        as_cuda_type(stop_status->get_data()),
        as_cuda_type(device_storage->get_data()));

    exec->get_master()->copy_from(exec.get(), 2,
                                  device_storage->get_const_data(), tmp);
    *all_converged = tmp[0];
    *one_changed = tmp[1];
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_RESIDUAL_NORM_REDUCTION_KERNEL);
