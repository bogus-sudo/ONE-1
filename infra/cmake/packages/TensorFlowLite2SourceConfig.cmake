function(_TensorFlowLite2Source_import platform)
    if(NOT DOWNLOAD_TENSORFLOW_LITE2)
        set(TensorFlowLite2Source_FOUND FALSE PARENT_SCOPE)
        return()
    endif(NOT DOWNLOAD_TENSORFLOW_LITE2)

    nnas_include(ExternalSourceTools)
    nnas_include(OptionTools)

    envoption(TENSORFLOW_LITE2_URL https://github.com/tensorflow/tensorflow/archive/v2.2.0-rc0.tar.gz)
    ExternalSource_Get("tensorflow2-${platform}" ${DOWNLOAD_TENSORFLOW_LITE2} ${TENSORFLOW_LITE2_URL})


    set(TensorFlowLite2Source_DIR ${tensorflow2-${platform}_SOURCE_DIR} PARENT_SCOPE)
    set(TensorFlowLite2Source_FOUND ${tensorflow2-${platform}_SOURCE_GET} PARENT_SCOPE)
endfunction(_TensorFlowLite2Source_import)

set(platform ${TARGET_ARCH}.${TARGET_OS})
_TensorFlowLite2Source_import(${platform})
