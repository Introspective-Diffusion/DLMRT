#include <torch/extension.h>

namespace dlmrt {
void gate_up_swiglu(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, bool pdl, int cluster, int mode,
                    bool round_silu);
}  // namespace dlmrt

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gate_up_swiglu", &dlmrt::gate_up_swiglu);
}
