function(_AbslSource_import)
  nnas_include(ExternalSourceTools)
  nnas_include(OptionTools)

  # NOTE The following URL comes from TensorFlow 1.12
  envoption(EXTERNAL_DOWNLOAD_SERVER "https://github.com")
  set(ABSL_URL ${EXTERNAL_DOWNLOAD_SERVER}/abseil/abseil-cpp/archive/389ec3f906f018661a5308458d623d01f96d7b23.tar.gz)
  ExternalSource_Get("absl" ${DOWNLOAD_ABSL} ${ABSL_URL})

  set(AbslSource_DIR ${absl_SOURCE_DIR} PARENT_SCOPE)
  set(AbslSource_FOUND ${absl_SOURCE_GET} PARENT_SCOPE)
endfunction(_AbslSource_import)

_AbslSource_import()
