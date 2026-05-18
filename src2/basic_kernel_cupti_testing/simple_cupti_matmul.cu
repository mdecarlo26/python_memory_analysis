// simple_cupti_matmul.cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cupti.h>

// ================================================================
// Simple matrix multiply kernel: C = A(MxK) * B(KxN)
// ================================================================
__global__ void matmul(const float *A, const float *B, float *C,
                       int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0..M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0..N-1

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ================================================================
// Error checking helpers
// ================================================================
#define CUDA_CALL(expr) do {                                            \
    cudaError_t _err = (expr);                                          \
    if (_err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                 \
                #expr, __FILE__, __LINE__, cudaGetErrorString(_err));   \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

#define CUPTI_CALL(expr) do {                                           \
    CUptiResult _status = (expr);                                       \
    if (_status != CUPTI_SUCCESS) {                                     \
        const char *errstr = NULL;                                      \
        cuptiGetResultString(_status, &errstr);                         \
        fprintf(stderr, "CUPTI error %s at %s:%d: %s\n",                \
                #expr, __FILE__, __LINE__,                              \
                errstr ? errstr : "(unknown)");                         \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

static double bytes_to_mib(unsigned long long b)
{
    return (double)b / (1024.0 * 1024.0);
}

// ================================================================
// Global counters for activity stats
// ================================================================

// Explicit cudaMemcpy activity (host ↔ device, etc.)
static unsigned long long g_bytes_htod   = 0;
static unsigned long long g_bytes_dtoh   = 0;
static unsigned long long g_bytes_dtod   = 0;
static unsigned long long g_bytes_htoh   = 0;
static unsigned long long g_bytes_other  = 0;

// Unified memory migration activity (accumulated deltas, bytes)
static unsigned long long g_um_htod      = 0;
static unsigned long long g_um_dtoh      = 0;
static unsigned long long g_um_dtod      = 0;
static unsigned long long g_um_other     = 0;

// Unified memory page fault counts (deltas, counts)
static unsigned long long g_um_cpu_pf    = 0;  // CPU page faults
static unsigned long long g_um_gpu_pf    = 0;  // GPU page faults

// Last raw UM counter values, for delta computation
static unsigned long long g_um_last_htod      = 0;
static unsigned long long g_um_last_dtoh      = 0;
static unsigned long long g_um_last_dtod      = 0;
static unsigned long long g_um_last_other     = 0;
static unsigned long long g_um_last_cpu_pf    = 0;
static unsigned long long g_um_last_gpu_pf    = 0;

// Flags to know if we’ve seen at least one sample for each counter
static int g_um_seen_htod      = 0;
static int g_um_seen_dtoh      = 0;
static int g_um_seen_dtod      = 0;
static int g_um_seen_other     = 0;
static int g_um_seen_cpu_pf    = 0;
static int g_um_seen_gpu_pf    = 0;

// ================================================================
// CUPTI Activity callbacks
// ================================================================
static void CUPTIAPI bufferRequested(uint8_t **buffer,
                                     size_t *size,
                                     size_t *maxNumRecords)
{
    const size_t BUF_SIZE = 32 * 1024;
    *size = BUF_SIZE;
    *buffer = (uint8_t*)malloc(BUF_SIZE);
    *maxNumRecords = 0; // no internal limit
}

// Handle each individual activity record
static void handleActivityRecord(const CUpti_Activity *record)
{
    // -------- Explicit cudaMemcpy activity --------
    if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
        const CUpti_ActivityMemcpy *m =
            (const CUpti_ActivityMemcpy*)record;

        switch (m->copyKind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            g_bytes_htod  += m->bytes;
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            g_bytes_dtoh  += m->bytes;
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            g_bytes_dtod  += m->bytes;
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            g_bytes_htoh  += m->bytes;
            break;
        default:
            g_bytes_other += m->bytes;
            break;
        }
    }

    // -------- Unified memory migration counters & page faults --------
    else if (record->kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
        const CUpti_ActivityUnifiedMemoryCounter *um =
            (const CUpti_ActivityUnifiedMemoryCounter*)record;

        unsigned long long curr = um->value;
        unsigned long long diff = 0;

        switch (um->counterKind) {
        // Bytes: Host -> Device
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
            if (g_um_seen_htod) {
                if (curr >= g_um_last_htod)
                    diff = curr - g_um_last_htod;
            } else {
                g_um_seen_htod = 1;
            }
            g_um_last_htod = curr;
            g_um_htod += diff;
            break;

        // Bytes: Device -> Host
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            if (g_um_seen_dtoh) {
                if (curr >= g_um_last_dtoh)
                    diff = curr - g_um_last_dtoh;
            } else {
                g_um_seen_dtoh = 1;
            }
            g_um_last_dtoh = curr;
            g_um_dtoh += diff;
            break;

        // Bytes: Device <-> Device
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD:
            if (g_um_seen_dtod) {
                if (curr >= g_um_last_dtod)
                    diff = curr - g_um_last_dtod;
            } else {
                g_um_seen_dtod = 1;
            }
            g_um_last_dtod = curr;
            g_um_dtod += diff;
            break;

        // CPU page faults (count, not bytes)
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
            if (g_um_seen_cpu_pf) {
                if (curr >= g_um_last_cpu_pf)
                    diff = curr - g_um_last_cpu_pf;
            } else {
                g_um_seen_cpu_pf = 1;
            }
            g_um_last_cpu_pf = curr;
            g_um_cpu_pf += diff;
            break;

        // GPU page faults (count, not bytes)
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
            if (g_um_seen_gpu_pf) {
                if (curr >= g_um_last_gpu_pf)
                    diff = curr - g_um_last_gpu_pf;
            } else {
                g_um_seen_gpu_pf = 1;
            }
            g_um_last_gpu_pf = curr;
            g_um_gpu_pf += diff;
            break;

        // Everything else
        default:
            if (g_um_seen_other) {
                if (curr >= g_um_last_other)
                    diff = curr - g_um_last_other;
            } else {
                g_um_seen_other = 1;
            }
            g_um_last_other = curr;
            g_um_other += diff;
            break;
        }
    }
    // If your CUDA version only has UNIFIED_MEMORY_COUNTER2/3,
    // swap struct and enum names accordingly.
}

static void CUPTIAPI bufferCompleted(CUcontext /*ctx*/,
                                     uint32_t /*streamId*/,
                                     uint8_t *buffer,
                                     size_t size,
                                     size_t /*validSize*/)
{
    CUpti_Activity *record = NULL;
    CUptiResult status;

    do {
        status = cuptiActivityGetNextRecord(buffer, size, &record);
        if (status == CUPTI_SUCCESS && record) {
            handleActivityRecord(record);
        }
    } while (status == CUPTI_SUCCESS);

    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED &&
        status != CUPTI_SUCCESS) {
        const char *errstr = NULL;
        cuptiGetResultString(status, &errstr);
        fprintf(stderr, "CUPTI error in cuptiActivityGetNextRecord: %s\n",
                errstr ? errstr : "(unknown)");
    }

    size_t dropped = 0;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(NULL, 0, &dropped));
    if (dropped > 0) {
        fprintf(stderr, "CUPTI WARNING: dropped %zu activity records\n", dropped);
    }

    free(buffer);
}

// ---------------------------------------------------------
// Configure unified memory counters, then enable UM activity
// ---------------------------------------------------------
static void configure_unified_memory_counters()
{
    int dev = 0;
    CUDA_CALL(cudaGetDevice(&dev));

    // We’ll enable 5 counters:
    //  - Bytes HtoD, DtoH, DtOD
    //  - CPU page faults, GPU page faults
    CUpti_ActivityUnifiedMemoryCounterConfig cfg[5];
    memset(cfg, 0, sizeof(cfg));

    // Scope: "single process, this device"
    CUpti_ActivityUnifiedMemoryCounterScope scope =
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;

    // Bytes: Host -> Device
    cfg[0].scope    = scope;
    cfg[0].kind     = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    cfg[0].deviceId = (uint32_t)dev;
    cfg[0].enable   = 1;

    // Bytes: Device -> Host
    cfg[1].scope    = scope;
    cfg[1].kind     = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    cfg[1].deviceId = (uint32_t)dev;
    cfg[1].enable   = 1;

    // Bytes: Device <-> Device
    cfg[2].scope    = scope;
    cfg[2].kind     = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD;
    cfg[2].deviceId = (uint32_t)dev;
    cfg[2].enable   = 1;

    // CPU page fault count
    cfg[3].scope    = scope;
    cfg[3].kind     = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT;
    cfg[3].deviceId = (uint32_t)dev; // ignored for CPU PF, but fine
    cfg[3].enable   = 1;

    // GPU page faults
    cfg[4].scope    = scope;
    cfg[4].kind     = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT;
    cfg[4].deviceId = (uint32_t)dev;
    cfg[4].enable   = 1;

    // Configure BEFORE enabling the UM activity kind
    CUPTI_CALL(cuptiActivityConfigureUnifiedMemoryCounter(cfg, 5));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
}

// Start collecting CUPTI activity
static void cupti_start()
{
    // Reset all counters
    g_bytes_htod = g_bytes_dtoh = g_bytes_dtod = g_bytes_htoh = g_bytes_other = 0;

    g_um_htod = g_um_dtoh = g_um_dtod = g_um_other = 0;
    g_um_cpu_pf = g_um_gpu_pf = 0;

    g_um_last_htod = g_um_last_dtoh = g_um_last_dtod = g_um_last_other = 0;
    g_um_last_cpu_pf = g_um_last_gpu_pf = 0;

    g_um_seen_htod = g_um_seen_dtoh = g_um_seen_dtod = g_um_seen_other = 0;
    g_um_seen_cpu_pf = g_um_seen_gpu_pf = 0;

    // Make sure CUDA context exists before configuring CUPTI UM
    CUDA_CALL(cudaFree(0)); // forces context creation if not already there

    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Explicit memcpy traffic
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));

    // Unified memory counters: configure first, then enable
    configure_unified_memory_counters();
}

// Stop and flush CUPTI activity
static void cupti_stop_and_flush()
{
    CUPTI_CALL(cuptiActivityFlushAll(0));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
}

// Pretty printing
static void print_memcpy_summary()
{
    unsigned long long total =
        g_bytes_htod + g_bytes_dtoh + g_bytes_dtod + g_bytes_htoh + g_bytes_other;

    printf("============================================================\n");
    printf("CUPTI Report: Explicit cudaMemcpy Traffic\n");
    printf("------------------------------------------------------------\n");
    printf("  Host -> Device (HtoD):  %10llu bytes (%.3f MiB)\n",
           g_bytes_htod,  bytes_to_mib(g_bytes_htod));
    printf("  Device -> Host (DtoH):  %10llu bytes (%.3f MiB)\n",
           g_bytes_dtoh,  bytes_to_mib(g_bytes_dtoh));
    printf("  Device -> Device(DtoD): %10llu bytes (%.3f MiB)\n",
           g_bytes_dtod,  bytes_to_mib(g_bytes_dtod));
    printf("  Host  -> Host (HtoH):   %10llu bytes (%.3f MiB)\n",
           g_bytes_htoh,  bytes_to_mib(g_bytes_htoh));
    printf("  Other kinds:            %10llu bytes (%.3f MiB)\n",
           g_bytes_other, bytes_to_mib(g_bytes_other));
    printf("------------------------------------------------------------\n");
    printf("  Total explicit memcpy:  %10llu bytes (%.3f MiB)\n",
           total, bytes_to_mib(total));
    printf("============================================================\n\n");
}

static void print_unified_memory_summary()
{
    unsigned long long total_bytes =
        g_um_htod + g_um_dtoh + g_um_dtod + g_um_other;

    printf("============================================================\n");
    printf("CUPTI Report: Unified Memory Migration + Page Faults\n");
    printf("  (On-demand page migration between host and device(s))\n");
    printf("------------------------------------------------------------\n");
    printf("  UM bytes Host -> Device:    %10llu (%.3f MiB)\n",
           g_um_htod,  bytes_to_mib(g_um_htod));
    printf("  UM bytes Device -> Host:    %10llu (%.3f MiB)\n",
           g_um_dtoh,  bytes_to_mib(g_um_dtoh));
    printf("  UM bytes Device <-> Device: %10llu (%.3f MiB)\n",
           g_um_dtod,  bytes_to_mib(g_um_dtod));
    printf("  UM bytes (other counters):  %10llu (%.3f MiB-equivalent)\n",
           g_um_other, bytes_to_mib(g_um_other));
    printf("------------------------------------------------------------\n");
    printf("  Total UM bytes (tracked):   %10llu (%.3f MiB-equivalent)\n",
           total_bytes, bytes_to_mib(total_bytes));
    printf("\n");
    printf("  UM CPU page faults:         %10llu\n", g_um_cpu_pf);
    printf("  UM GPU page faults:         %10llu\n", g_um_gpu_pf);
    printf("============================================================\n\n");
}

// ================================================================
// Main
// ================================================================
int main(int argc, char **argv)
{
    // Bigger matrices by default: 1024 x 1024
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int K = (argc > 2) ? atoi(argv[2]) : 1024;
    int N = (argc > 3) ? atoi(argv[3]) : 1024;

    printf("Running simple matmul with CUPTI unified memory profiling\n");
    printf("  Dimensions: C(MxN) = A(MxK) * B(KxN)\n");
    printf("  M = %d, K = %d, N = %d\n\n", M, K, N);

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *A = NULL;
    float *B = NULL;
    float *C = NULL;

    // Init CUDA + CUPTI collection before allocations to see UM traffic
    CUDA_CALL(cudaSetDevice(0));
    cupti_start();

    // Use unified memory for A, B, C
    CUDA_CALL(cudaMallocManaged(&A, sizeA));
    CUDA_CALL(cudaMallocManaged(&B, sizeB));
    CUDA_CALL(cudaMallocManaged(&C, sizeC));

    // Initialize on the host (will cause pages to be mapped to host)
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 1.0f;
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    // No explicit cudaMemcpy() needed: UM will migrate pages on demand.

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);

    // Launch single simple kernel
    matmul<<<blocks, threads>>>(A, B, C, M, N, K);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Touch result on host to ensure UM migration back if needed
    double checksum = 0.0;
    for (int i = 0; i < M * N; ++i) checksum += C[i];

    // Stop & flush CUPTI, then print out stats
    cupti_stop_and_flush();
    print_memcpy_summary();          // likely zero (no cudaMemcpy calls)
    print_unified_memory_summary();  // UM bytes + CPU/GPU page faults

    // Print only a small corner so we don't spam stdout
    int maxRowsToPrint = (M < 4) ? M : 4;
    int maxColsToPrint = (N < 4) ? N : 4;

    printf("Top-left corner of result matrix C (up to 4x4):\n");
    for (int i = 0; i < maxRowsToPrint; ++i) {
        for (int j = 0; j < maxColsToPrint; ++j) {
            printf("%8.2f ", C[i * N + j]);
        }
        if (maxColsToPrint < N) printf(" ...");
        printf("\n");
    }
    if (maxRowsToPrint < M) {
        printf("...\n");
    }

    printf("\nResult checksum (for sanity): %.4f\n", checksum);

    CUDA_CALL(cudaFree(A));
    CUDA_CALL(cudaFree(B));
    CUDA_CALL(cudaFree(C));

    return 0;
}
