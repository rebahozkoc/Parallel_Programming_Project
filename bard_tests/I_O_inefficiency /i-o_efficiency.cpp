
//commit ac24746

Status Writer::close_files(FragmentMetadata* meta) const {
  // Close attribute and dimension files
  for (const auto& it : buffers_) {
    const auto& name = it.first;
    RETURN_NOT_OK(storage_manager_->close_file(meta->uri(name)));
    if (array_schema_->var_size(name))
      RETURN_NOT_OK(storage_manager_->close_file(meta->var_uri(name)));
    if (array_schema_->is_nullable(name))
      RETURN_NOT_OK(storage_manager_->close_file(meta->validity_uri(name)));
  }

  return Status::Ok();
}