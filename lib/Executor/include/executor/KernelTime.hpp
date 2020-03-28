#pragma once
namespace executor {
typedef struct {
  float upload{0};
  float download{0};
  float launch{0};
  float total{0};
} KernelTime;
} // namespace executor