#include "tfl_gpu_backend_tests.h"

void TflGpuBackendTest::SetUp() {
  old_value_of_BACKENDS_env_var = getenv("BACKENDS");
  setenv("BACKENDS", "cpu;tfl_gpu", 1);
  old_value_of_EXECUTOR_env_var = getenv("EXECUTOR");
  setenv("EXECUTOR", "Linear", 1);
  platform = "x86_64-linux";
}