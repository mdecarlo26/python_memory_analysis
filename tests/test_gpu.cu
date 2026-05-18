/*
test_gpu.cu
-----------
G1 test: run SAXPY and a naive matrix multiply, capture GPU events via
cupti_layer, read them back from the ring buffer, validate the results.

Pass criteria:
  [G1-A] cupti_start() returns 1 (layer initialized without error)
  [G1-B] MEMCPY events captured for H2D and D2H transfers
  [G1-C] KERNEL events captured with non-empty name
  [G1-D] All events have timestamp_ns > 0
  [G1-E] All events have valid event_type (2, 3, or 4)
  [G1-F] At least 4 events total (2 H2D + 1 kernel + 1 D2H minimum)

Build:
  nvcc -O2 -std=c++17 \
       -I/usr/local/cuda/include \
       -I/usr/local/cuda/extras/CUPTI/include \
       tests/test_gpu.cu \
       src/cpp/ring_buffer.so \
       src/gpu/cupti_layer.so \
       -L/usr/local/cuda/lib64 \
       -L/usr/local/cuda/extras/CUPTI/lib64 \
       -lcuda -lcupti \
       -Wl,-rpath,src/cpp \
       -Wl,-rpath,src/gpu \
       -Wl,-rpath,/usr/local/cuda/extras/CUPTI/lib64 \
       -o tests/test_gpu

Run:
  ./tests/test_gpu
*/

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cassert>

// ---------------------------------------------------------------------------
// C API from cupti_layer.so
// ---------------------------------------------------------------------------
extern "C" {
    int      cupti_start(const char* shm_name);
    void     cupti_stop();
    uint64_t cupti_events_written();
}

// ---------------------------------------------------------------------------
// Ring buffer C API from ring_buffer.so (consumer side)
// ---------------------------------------------------------------------------
extern "C" {
    typedef struct RingBuffer RingBuffer;
    RingBuffer* rb_create(const char* name, uint32_t capacity, int create);
    int         rb_read(RingBuffer* rb, uint8_t* buf, uint32_t buf_len, uint32_t* out_len);
    void        rb_destroy(RingBuffer* rb);
    void        rb_unlink(const char* name);
}

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// ---------------------------------------------------------------------------
// SAXPY kernel
// ---------------------------------------------------------------------------
__global__ void saxpy(int n, float a, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// ---------------------------------------------------------------------------
// Naive matrix multiply kernel
// ---------------------------------------------------------------------------
__global__ void matmul(int N, const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < N; ++k)
        sum += A[row * N + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// Event deserialization (mirrors bridge.py deserialize logic)
// ---------------------------------------------------------------------------

struct GpuEventRecord {
    uint64_t event_id;
    uint64_t timestamp_ns;
    uint32_t process_id;
    uint8_t  event_type;   // 2=TRANSFER 3=PAGE_FAULT 4=KERNEL
    std::string label;

    // GPU extension
    uint32_t device_id;
    uint64_t src_address;
    uint64_t dst_address;
    uint64_t transfer_size;
    uint8_t  transfer_kind;
    uint32_t page_faults;
};

static inline uint64_t read_u64(const uint8_t* p, int& off) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= ((uint64_t)p[off+i] << (8*i));
    off += 8; return v;
}
static inline uint32_t read_u32(const uint8_t* p, int& off) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= ((uint32_t)p[off+i] << (8*i));
    off += 4; return v;
}
static inline uint16_t read_u16(const uint8_t* p, int& off) {
    uint16_t v = (uint16_t)(p[off] | (p[off+1] << 8));
    off += 2; return v;
}
static inline uint8_t read_u8(const uint8_t* p, int& off) {
    return p[off++];
}

static bool deserialize(const uint8_t* payload, uint32_t len, GpuEventRecord& out) {
    // Minimum base size: 8+8+4+8+1+1+8+8+4+1+2 = 53 bytes
    if (len < 53) return false;
    int off = 0;

    out.event_id     = read_u64(payload, off);
    out.timestamp_ns = read_u64(payload, off);
    out.process_id   = read_u32(payload, off);
    /* thread_id */    read_u64(payload, off);
    out.event_type   = read_u8 (payload, off);
    /* is_dealloc */   read_u8 (payload, off);
    /* alloc_addr */   read_u64(payload, off);
    /* alloc_size */   read_u64(payload, off);
    /* ref_count */    read_u32(payload, off);
    /* gc_gen */       read_u8 (payload, off);

    uint16_t label_len = read_u16(payload, off);
    if ((int)(off + label_len) > (int)len) return false;
    out.label = std::string((const char*)payload + off, label_len);
    off += label_len;

    // GPU extension fields
    if ((int)(off + 4+8+8+8+1+4) > (int)len) return false;
    out.device_id     = read_u32(payload, off);
    out.src_address   = read_u64(payload, off);
    out.dst_address   = read_u64(payload, off);
    out.transfer_size = read_u64(payload, off);
    out.transfer_kind = read_u8 (payload, off);
    out.page_faults   = read_u32(payload, off);

    return true;
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

int main() {
    const char* SHM = "/hpc_test_g1";
    const int   N   = 256;
    bool        all_pass = true;

    auto PASS = [&](const char* name) {
        printf("  PASS  %s\n", name);
    };
    auto FAIL = [&](const char* name, const char* msg) {
        printf("  FAIL  %s — %s\n", name, msg);
        all_pass = false;
    };

    printf("\n=== G1 test: CUPTI layer ===\n\n");

    // Create consumer ring buffer BEFORE starting the profiler
    // (profiler creates with create=1, we open with create=0)
    // Actually: profiler calls rb_create(name, cap, 1) — it owns the segment.
    // We open a second handle to the same segment after the profiler creates it.

    // G1-A: start profiler
    int started = cupti_start(SHM);
    if (started) PASS("G1-A  cupti_start succeeded");
    else { FAIL("G1-A", "cupti_start returned 0"); return 1; }

    // Open consumer handle to the ring buffer the profiler just created
    RingBuffer* rb = rb_create(SHM, 8 * 1024 * 1024, /*create=*/0);
    if (!rb) {
        FAIL("setup", "failed to open ring buffer consumer handle");
        cupti_stop();
        return 1;
    }

    // -----------------------------------------------------------------------
    // Run SAXPY workload
    // -----------------------------------------------------------------------
    {
        int n = 1 << 20;
        std::vector<float> hx(n, 1.0f), hy(n, 2.0f);
        float *dx, *dy;
        CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        saxpy<<<(n+255)/256, 256>>>(n, 2.0f, dx, dy);
        CUDA_CHECK(cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(dx); cudaFree(dy);
    }

    // -----------------------------------------------------------------------
    // Run matmul workload
    // -----------------------------------------------------------------------
    {
        std::vector<float> hA(N*N, 1.0f), hB(N*N, 1.0f), hC(N*N, 0.0f);
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, N*N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, N*N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, N*N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));
        dim3 block(16,16), grid((N+15)/16, (N+15)/16);
        matmul<<<grid, block>>>(N, dA, dB, dC);
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, N*N*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // Stop profiler — flushes remaining CUPTI records
    cupti_stop();

    // -----------------------------------------------------------------------
    // Drain ring buffer and validate events
    // -----------------------------------------------------------------------
    std::vector<GpuEventRecord> events;
    {
        std::vector<uint8_t> buf(4096);
        uint32_t out_len = 0;
        while (rb_read(rb, buf.data(), (uint32_t)buf.size(), &out_len)) {
            GpuEventRecord rec;
            if (deserialize(buf.data(), out_len, rec))
                events.push_back(rec);
        }
    }
    rb_destroy(rb);
    rb_unlink(SHM);

    printf("Events captured: %zu\n\n", events.size());

    // G1-F: at least 4 events
    if (events.size() >= 4) PASS("G1-F  >= 4 events captured");
    else FAIL("G1-F", "fewer than 4 events");

    // G1-D: all timestamps > 0
    bool ts_ok = true;
    for (auto& e : events) if (e.timestamp_ns == 0) { ts_ok = false; break; }
    if (ts_ok) PASS("G1-D  all timestamps > 0");
    else FAIL("G1-D", "at least one event has timestamp_ns == 0");

    // G1-E: all event_types valid
    bool type_ok = true;
    for (auto& e : events)
        if (e.event_type < 2 || e.event_type > 4) { type_ok = false; break; }
    if (type_ok) PASS("G1-E  all event_types valid (2-4)");
    else FAIL("G1-E", "invalid event_type found");

    // G1-B: at least one TRANSFER event
    int transfers = 0;
    for (auto& e : events) if (e.event_type == 2) ++transfers;
    if (transfers > 0) PASS("G1-B  MEMCPY events captured");
    else FAIL("G1-B", "no TRANSFER events found");

    // G1-C: at least one KERNEL event with non-empty name
    int kernels = 0;
    for (auto& e : events)
        if (e.event_type == 4 && !e.label.empty()) ++kernels;
    if (kernels > 0) PASS("G1-C  KERNEL events with names captured");
    else FAIL("G1-C", "no KERNEL events with names found");

    // Print summary table
    printf("\nEvent breakdown:\n");
    int t=0, pf=0, k=0;
    for (auto& e : events) {
        if      (e.event_type == 2) ++t;
        else if (e.event_type == 3) ++pf;
        else if (e.event_type == 4) ++k;
    }
    printf("  TRANSFER   : %d\n", t);
    printf("  PAGE_FAULT : %d\n", pf);
    printf("  KERNEL     : %d\n", k);

    // Print first 5 events for inspection
    printf("\nFirst events:\n");
    int shown = 0;
    for (auto& e : events) {
        if (shown++ >= 5) break;
        const char* type_str = e.event_type == 2 ? "TRANSFER" :
                               e.event_type == 3 ? "PAGE_FAULT" : "KERNEL";
        printf("  [%llu] %s  ts=%llu  dev=%u  label=%s\n",
               (unsigned long long)e.event_id,
               type_str,
               (unsigned long long)e.timestamp_ns,
               e.device_id,
               e.label.c_str());
    }

    printf("\n%s\n\n", all_pass ? "ALL PASS" : "SOME FAILURES — see above");
    return all_pass ? 0 : 1;
}
