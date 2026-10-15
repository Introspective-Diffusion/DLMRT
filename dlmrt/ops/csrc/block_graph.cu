// The per-block CUDA graph: begin -> WHILE(remaining && step < limit){round} -> commit.
// The host launches it once per block; every denoise round is decided on the device.
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// CUDA 13 added per-edge data to cudaGraphAddNode.
static cudaError_t add_node(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps, size_t count,
                            cudaGraphNodeParams* params) {
#if CUDART_VERSION >= 13000
    return cudaGraphAddNode(node, graph, deps, nullptr, count, params);
#else
    return cudaGraphAddNode(node, graph, deps, count, params);
#endif
}

__global__ void set_condition(cudaGraphConditionalHandle handle, const int* remaining, const int* step, int limit) {
    cudaGraphSetConditional(handle, *remaining > 0 && *step < limit);
}

class BlockGraph {
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;

public:
    BlockGraph(uintptr_t begin, uintptr_t round, uintptr_t commit, uintptr_t remaining_ptr,
               uintptr_t step_ptr, int limit) {
        try {
            check(cudaGraphCreate(&graph_, 0));
            cudaGraphConditionalHandle handle;
            check(cudaGraphConditionalHandleCreate(&handle, graph_, 0, 0));
            auto remaining = reinterpret_cast<int*>(remaining_ptr);
            auto step = reinterpret_cast<int*>(step_ptr);
            void* args[] = {&handle, &remaining, &step, &limit};
            cudaKernelNodeParams params{};
            params.func = reinterpret_cast<void*>(set_condition);
            params.gridDim = dim3(1);
            params.blockDim = dim3(1);
            params.kernelParams = args;

            cudaGraphNode_t first, entry, loop, body_round, body_tail, last;
            check(cudaGraphAddChildGraphNode(&first, graph_, nullptr, 0, reinterpret_cast<cudaGraph_t>(begin)));
            check(cudaGraphAddKernelNode(&entry, graph_, &first, 1, &params));
            cudaGraphNodeParams conditional{};
            conditional.type = cudaGraphNodeTypeConditional;
            conditional.conditional.handle = handle;
            conditional.conditional.type = cudaGraphCondTypeWhile;
            conditional.conditional.size = 1;
            check(add_node(&loop, graph_, &entry, 1, &conditional));
            cudaGraph_t body = conditional.conditional.phGraph_out[0];
            check(cudaGraphAddChildGraphNode(&body_round, body, nullptr, 0, reinterpret_cast<cudaGraph_t>(round)));
            check(cudaGraphAddKernelNode(&body_tail, body, &body_round, 1, &params));
            check(cudaGraphAddChildGraphNode(&last, graph_, &loop, 1, reinterpret_cast<cudaGraph_t>(commit)));
            check(cudaGraphInstantiate(&exec_, graph_, 0));
        } catch (...) {
            if (exec_) cudaGraphExecDestroy(exec_);
            if (graph_) cudaGraphDestroy(graph_);
            throw;
        }
    }
    ~BlockGraph() {
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
    }
    void replay(uintptr_t stream) { check(cudaGraphLaunch(exec_, reinterpret_cast<cudaStream_t>(stream))); }
};

// begin -> WHILE(remaining_0 && step_0 < limit_0){body_0} -> WHILE(...){body_1} -> ... -> tail (none if 0).
class LoopGraph {
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;

public:
    // loops: (body graph, remaining pointer, step pointer, limit) in order
    LoopGraph(uintptr_t begin, std::vector<std::tuple<uintptr_t, uintptr_t, uintptr_t, int>> loops, uintptr_t tail) {
        try {
            check(cudaGraphCreate(&graph_, 0));
            cudaGraphNode_t last;
            check(cudaGraphAddChildGraphNode(&last, graph_, nullptr, 0, reinterpret_cast<cudaGraph_t>(begin)));
            for (auto& [body_graph, remaining_ptr, step_ptr, limit_value] : loops) {
                cudaGraphConditionalHandle handle;
                check(cudaGraphConditionalHandleCreate(&handle, graph_, 0, 0));
                auto remaining = reinterpret_cast<int*>(remaining_ptr);
                auto step = reinterpret_cast<int*>(step_ptr);
                int limit = limit_value;
                void* args[] = {&handle, &remaining, &step, &limit};
                cudaKernelNodeParams params{};
                params.func = reinterpret_cast<void*>(set_condition);
                params.gridDim = dim3(1);
                params.blockDim = dim3(1);
                params.kernelParams = args;
                cudaGraphNode_t entry, loop, body_round, body_tail;
                check(cudaGraphAddKernelNode(&entry, graph_, &last, 1, &params));
                cudaGraphNodeParams conditional{};
                conditional.type = cudaGraphNodeTypeConditional;
                conditional.conditional.handle = handle;
                conditional.conditional.type = cudaGraphCondTypeWhile;
                conditional.conditional.size = 1;
                check(add_node(&loop, graph_, &entry, 1, &conditional));
                cudaGraph_t body = conditional.conditional.phGraph_out[0];
                check(cudaGraphAddChildGraphNode(&body_round, body, nullptr, 0, reinterpret_cast<cudaGraph_t>(body_graph)));
                check(cudaGraphAddKernelNode(&body_tail, body, &body_round, 1, &params));
                last = loop;
            }
            if (tail) {
                cudaGraphNode_t end;
                check(cudaGraphAddChildGraphNode(&end, graph_, &last, 1, reinterpret_cast<cudaGraph_t>(tail)));
            }
            check(cudaGraphInstantiate(&exec_, graph_, 0));
        } catch (...) {
            if (exec_) cudaGraphExecDestroy(exec_);
            if (graph_) cudaGraphDestroy(graph_);
            throw;
        }
    }
    ~LoopGraph() {
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
    }
    void replay(uintptr_t stream) { check(cudaGraphLaunch(exec_, reinterpret_cast<cudaStream_t>(stream))); }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<BlockGraph>(m, "BlockGraph")
        .def(pybind11::init<uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, int>())
        .def("replay", &BlockGraph::replay);
    pybind11::class_<LoopGraph>(m, "LoopGraph")
        .def(pybind11::init<uintptr_t, std::vector<std::tuple<uintptr_t, uintptr_t, uintptr_t, int>>, uintptr_t>())
        .def("replay", &LoopGraph::replay);
}
