// advanced_cupti_matmul.cu
// Demo:
//  - Simple matmul kernel with Unified Memory
//  - CUPTI activity tracing:
//      * Explicit cudaMemcpy traffic
//      * Unified Memory bytes + page faults (global + per-array)
//      * PC sampling + global access record counts, when supported
//
// On GPUs where PC sampling / global access aren't supported, those
// features are disabled gracefully and you'll see a warning + zeros.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>          // for ULLONG_MAX

#include <cuda.h>            // for cuCtxGetCurrent
#include <cuda_runtime.h>
#include <cupti.h>

// ======================================================================
// Simple matrix multiply kernel: C = A(MxK) * B(KxN)
// ======================================================================
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

// ======================================================================
// Error checking helpers
// ======================================================================
#define CUDA_CALL(expr) do {                                                \
    cudaError_t _err = (expr);                                              \
    if (_err != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                     \
                #expr, __FILE__, __LINE__, cudaGetErrorString(_err));       \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

#define CU_CALL(expr) do {                                                  \
    CUresult _res = (expr);                                                 \
    if (_res != CUDA_SUCCESS) {                                             \
        const char *errstr = NULL;                                          \
        cuGetErrorString(_res, &errstr);                                    \
        fprintf(stderr, "CUDA driver error %s at %s:%d: %s\n",              \
                #expr, __FILE__, __LINE__, errstr ? errstr : "(unknown)");  \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

#define CUPTI_CALL(expr) do {                                               \
    CUptiResult _status = (expr);                                           \
    if (_status != CUPTI_SUCCESS) {                                         \
        const char *errstr = NULL;                                          \
        cuptiGetResultString(_status, &errstr);                             \
        fprintf(stderr, "CUPTI error %s at %s:%d: %s\n",                    \
                #expr, __FILE__, __LINE__,                                  \
                errstr ? errstr : "(unknown)");                             \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

static double bytes_to_mib(unsigned long long b)
{
    return (double)b / (1024.0 * 1024.0);
}

// ======================================================================
// Overflow-safe helpers
// ======================================================================
static inline void safe_range(uint64_t addr, uint64_t len,
                              uint64_t *out_start,
                              uint64_t *out_end)
{
    *out_start = addr;
    if (len > UINT64_MAX - addr) {
        // clamp if it would overflow
        *out_end = UINT64_MAX;
    } else {
        *out_end = addr + len;
    }
}

static inline void sat_add_u64(unsigned long long *acc,
                               unsigned long long inc)
{
    if (*acc > ULLONG_MAX - inc) {
        *acc = ULLONG_MAX; // saturate
    } else {
        *acc += inc;
    }
}

// ======================================================================
// Simple array registry so we can attribute UM traffic per-array
// ======================================================================
typedef struct {
    const char    *name;
    uint64_t       base;   // reinterpret_cast<uint64_t>(ptr)
    size_t         size;   // bytes
    // UM bytes by direction:
    unsigned long long um_bytes_htod;
    unsigned long long um_bytes_dtoh;
    unsigned long long um_bytes_dtod;
    // UM page faults:
    unsigned long long um_cpu_page_faults; // count of CPU PF events (records)
    unsigned long long um_gpu_page_faults; // sum of GPU PF "groups"
} ArrayInfo;

static ArrayInfo g_arrays[4];   // A, B, C, Other
static int       g_num_arrays = 0;

static void register_array(const char *name, void *ptr, size_t size)
{
    if (g_num_arrays >= (int)(sizeof(g_arrays)/sizeof(g_arrays[0]))) {
        fprintf(stderr, "Too many registered arrays, ignoring %s\n", name);
        return;
    }
    ArrayInfo *a = &g_arrays[g_num_arrays++];
    a->name = name;
    a->base = (uint64_t)ptr;
    a->size = size;
    a->um_bytes_htod = 0;
    a->um_bytes_dtoh = 0;
    a->um_bytes_dtod = 0;
    a->um_cpu_page_faults = 0;
    a->um_gpu_page_faults = 0;
}

static ArrayInfo* find_array_for_range(uint64_t addr, uint64_t len)
{
    uint64_t a0, a1;
    safe_range(addr, len, &a0, &a1);

    for (int i = 0; i < g_num_arrays; ++i) {
        uint64_t base, limit;
        // protect base + size as well
        safe_range(g_arrays[i].base, (uint64_t)g_arrays[i].size, &base, &limit);

        // Overlap check: [a0, a1) intersects [base, limit)
        if (a0 < limit && a1 > base) {
            return &g_arrays[i];
        }
    }
    return NULL;
}

// ======================================================================
// Global counters for activity stats
// ======================================================================

// Explicit cudaMemcpy activity
static unsigned long long g_bytes_htod   = 0;
static unsigned long long g_bytes_dtoh   = 0;
static unsigned long long g_bytes_dtod   = 0;
static unsigned long long g_bytes_htoh   = 0;
static unsigned long long g_bytes_other  = 0;

// Unified Memory migration totals (global)
static unsigned long long g_um_htod      = 0;
static unsigned long long g_um_dtoh      = 0;
static unsigned long long g_um_dtod      = 0;
static unsigned long long g_um_other_val = 0;

// Unified Memory page faults (global)
static unsigned long long g_um_cpu_page_faults = 0;  // count of records
static unsigned long long g_um_gpu_page_faults = 0;  // groups count

// PC sampling / global access activity counts
static unsigned long long g_pc_sampling_records   = 0;
static unsigned long long g_global_access_records = 0;

// Whether these activity kinds are actually enabled (device may not support them)
static int g_pc_sampling_enabled    = 0;
static int g_global_access_enabled  = 0;

// ======================================================================
// CUPTI activity callbacks
// ======================================================================
static void CUPTIAPI bufferRequested(uint8_t **buffer,
                                     size_t *size,
                                     size_t *maxNumRecords)
{
    const size_t BUF_SIZE = 32 * 1024;
    *size = BUF_SIZE;
    *buffer = (uint8_t*)malloc(BUF_SIZE);
    *maxNumRecords = 0; // no internal limit
}

static void handleActivityRecord(const CUpti_Activity *record)
{
    switch (record->kind) {
    // ----------------------------------------------------------
    // Explicit cudaMemcpy activity
    // ----------------------------------------------------------
    case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
        const CUpti_ActivityMemcpy *m =
            (const CUpti_ActivityMemcpy*)record;
        switch (m->copyKind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            sat_add_u64(&g_bytes_htod, m->bytes);
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            sat_add_u64(&g_bytes_dtoh, m->bytes);
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            sat_add_u64(&g_bytes_dtod, m->bytes);
            break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            sat_add_u64(&g_bytes_htoh, m->bytes);
            break;
        default:
            sat_add_u64(&g_bytes_other, m->bytes);
            break;
        }
        break;
    }

    // ----------------------------------------------------------
    // Unified Memory counters (bytes + page faults)
    // NOTE: using UnifiedMemoryCounter2; if your headers only have
    //       CUpti_ActivityUnifiedMemoryCounter, change the cast.
    // ----------------------------------------------------------
    case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
    {
        const CUpti_ActivityUnifiedMemoryCounter2 *um =
            (const CUpti_ActivityUnifiedMemoryCounter2*)record;

        CUpti_ActivityUnifiedMemoryCounterKind k = um->counterKind;
        uint64_t addr = um->address;
        uint64_t val  = um->value;   // semantics depend on counterKind

        ArrayInfo *arr = NULL;
        if (k == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD ||
            k == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH ||
            k == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD) {
            arr = find_array_for_range(addr, val);
            // If not one of A/B/C, dump into "Other" (last slot) if present
            if (!arr && g_num_arrays > 0) {
                arr = &g_arrays[g_num_arrays - 1];
            }
        }

        switch (k) {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
            sat_add_u64(&g_um_htod, val);
            if (arr) sat_add_u64(&arr->um_bytes_htod, val);
            break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
            sat_add_u64(&g_um_dtoh, val);
            if (arr) sat_add_u64(&arr->um_bytes_dtoh, val);
            break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD:
            sat_add_u64(&g_um_dtod, val);
            if (arr) sat_add_u64(&arr->um_bytes_dtod, val);
            break;

        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
            // For this counter kind, "value" is PC; each record is one CPU fault.
            sat_add_u64(&g_um_cpu_page_faults, 1);
            if (arr) sat_add_u64(&arr->um_cpu_page_faults, 1);
            break;

        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
            // "value" is number of page fault groups.
            sat_add_u64(&g_um_gpu_page_faults, val);
            if (arr) sat_add_u64(&arr->um_gpu_page_faults, val);
            break;

        default:
            sat_add_u64(&g_um_other_val, val);
            break;
        }
        break;
    }

    // ----------------------------------------------------------
    // PC sampling: just count records for this demo
    // ----------------------------------------------------------
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
    {
        //const CUpti_ActivityPCSampling3 *pc =
         //c   (const CUpti_ActivityPCSampling3*)record;
        //c(void)pc;
        //csat_add_u64(&g_pc_sampling_records, 1);
        //cbreak;
    }

    // ----------------------------------------------------------
    // Global access activity: just count records for this demo
    // ----------------------------------------------------------
    case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS:
    {
        //cconst CUpti_ActivityGlobalAccess3 *ga =
          //c  (const CUpti_ActivityGlobalAccess3*)record;
        //c(void)ga;
        //csat_add_u64(&g_global_access_records, 1);
        //cbreak;
    }

    default:
        // ignore other kinds
        break;
    }
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

// ======================================================================
// CUPTI setup / teardown
// ======================================================================
static void cupti_start()
{
    // Register buffer callbacks
    CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

    // Configure Unified Memory counters
    CUpti_ActivityUnifiedMemoryCounterConfig um_configs[5];
    memset(um_configs, 0, sizeof(um_configs));
    // Scope: "single process, this device"
    CUpti_ActivityUnifiedMemoryCounterScope scope =
        CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;

    um_configs[0].scope  = scope;
    um_configs[0].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
    um_configs[0].enable = 1;

    um_configs[1].scope  = scope;
    um_configs[1].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
    um_configs[1].enable = 1;

    um_configs[2].scope  = scope;
    um_configs[2].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD;
    um_configs[2].enable = 1;

    um_configs[3].scope  = scope;
    um_configs[3].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT;
    um_configs[3].enable = 1;

    um_configs[4].scope  = scope;
    um_configs[4].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT;
    um_configs[4].enable = 1;

    CUPTI_CALL(cuptiActivityConfigureUnifiedMemoryCounter(
        um_configs, (uint32_t)(sizeof(um_configs)/sizeof(um_configs[0]))));

    // Configure PC sampling for the current context (if supported)
    CUcontext ctx = NULL;
    CU_CALL(cuCtxGetCurrent(&ctx));
    if (ctx == NULL) {
        fprintf(stderr, "ERROR: cuCtxGetCurrent returned NULL (no active context)\n");
        exit(EXIT_FAILURE);
    }

    CUpti_ActivityPCSamplingConfig pcConfig;
    memset(&pcConfig, 0, sizeof(pcConfig));
    pcConfig.size            = sizeof(pcConfig);
    pcConfig.samplingPeriod2 = 10;  // 2^10 cycles per sample

    CUptiResult ps_cfg = cuptiActivityConfigurePCSampling(ctx, &pcConfig);
    if (ps_cfg == CUPTI_SUCCESS) {
        g_pc_sampling_enabled = 1;
    } else if (ps_cfg == CUPTI_ERROR_NOT_COMPATIBLE) {
        fprintf(stderr,
                "CUPTI: PC sampling is NOT compatible on this device/config; "
                "PC sampling disabled.\n");
        g_pc_sampling_enabled = 0;
    } else {
        // Any other error is fatal
        CUPTI_CALL(ps_cfg);
    }

    // Enable activity kinds
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));

    // PC SAMPLING
    if (g_pc_sampling_enabled) {
        CUptiResult ps_en = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_PC_SAMPLING);
        if (ps_en == CUPTI_SUCCESS) {
            // ok
        } else if (ps_en == CUPTI_ERROR_NOT_COMPATIBLE) {
            fprintf(stderr,
                    "CUPTI: PC sampling enable returned NOT_COMPATIBLE; "
                    "disabling PC sampling.\n");
            g_pc_sampling_enabled = 0;
        } else {
            CUPTI_CALL(ps_en);
        }
    }

    // GLOBAL ACCESS
    CUptiResult ga_en = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS);
    if (ga_en == CUPTI_SUCCESS) {
        g_global_access_enabled = 1;
    } else if (ga_en == CUPTI_ERROR_NOT_COMPATIBLE) {
        fprintf(stderr,
                "CUPTI: Global access activity is NOT compatible on this "
                "device/config; disabling global access activity.\n");
        g_global_access_enabled = 0;
    } else {
        CUPTI_CALL(ga_en);
    }
}

static void cupti_stop_and_flush()
{
    CUPTI_CALL(cuptiActivityFlushAll(0));

    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));

    if (g_pc_sampling_enabled) {
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_PC_SAMPLING));
    }
    if (g_global_access_enabled) {
        CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS));
    }
}

// ======================================================================
// Pretty printing helpers
// ======================================================================
// static void print_memcpy_summary()
// {
  //   unsigned long long total =
    //     g_bytes_htod + g_bytes_dtoh + g_bytes_dtod + g_bytes_htoh + g_bytes_other;

    // printf("============================================================\n");
    // printf("CUPTI Report: Explicit cudaMemcpy Traffic\n");
    // printf("------------------------------------------------------------\n");
    // printf("  Host -> Device (HtoD):  %12llu bytes (%.3f MiB)\n",
           // g_bytes_htod,  bytes_to_mib(g_bytes_htod));
    // printf("  Device -> Host (DtoH):  %12llu bytes (%.3f MiB)\n",
           // g_bytes_dtoh,  bytes_to_mib(g_bytes_dtoh));
    // printf("  Device -> Device(DtoD): %12llu bytes (%.3f MiB)\n",
           // g_bytes_dtod,  bytes_to_mib(g_bytes_dtod));
    // printf("  Host  -> Host (HtoH):   %12llu bytes (%.3f MiB)\n",
           // g_bytes_htoh,  bytes_to_mib(g_bytes_htoh));
    // printf("  Other kinds:            %12llu bytes (%.3f MiB)\n",
           // g_bytes_other, bytes_to_mib(g_bytes_other));
    // printf("------------------------------------------------------------\n");
    // printf("  Total explicit memcpy:  %12llu bytes (%.3f MiB)\n",
           // total, bytes_to_mib(total));
    // printf("============================================================\n\n");
// }

static void print_unified_memory_summary()
{
    unsigned long long total_bytes =
        g_um_htod + g_um_dtoh + g_um_dtod + g_um_other_val;

    printf("============================================================\n");
    printf("CUPTI Report: Unified Memory Migration + Page Faults\n");
    printf("------------------------------------------------------------\n");
    printf("  UM bytes Host -> Device:      %12llu (%.3f MiB)\n",
           g_um_htod,  bytes_to_mib(g_um_htod));
    printf("  UM bytes Device -> Host:      %12llu (%.3f MiB)\n",
           g_um_dtoh,  bytes_to_mib(g_um_dtoh));
    printf("  UM bytes Device <-> Device:   %12llu (%.3f MiB)\n",
           g_um_dtod,  bytes_to_mib(g_um_dtod));
    printf("  UM bytes (other counters):    %12llu (%.3f MiB)\n",
           g_um_other_val, bytes_to_mib(g_um_other_val));
    printf("------------------------------------------------------------\n");
    printf("  Total UM bytes (tracked):     %12llu (%.3f MiB)\n\n",
           total_bytes, bytes_to_mib(total_bytes));

    printf("  UM CPU page faults (records): %12llu\n", g_um_cpu_page_faults);
    printf("  UM GPU page fault groups:     %12llu\n", g_um_gpu_page_faults);
    printf("============================================================\n\n");
}

// static void print_array_unified_memory_summary()
// {
    // printf("============================================================\n");
    // printf("Per-array Unified Memory Migration + Page Faults\n");
    // printf("------------------------------------------------------------\n");
    // for (int i = 0; i < g_num_arrays; ++i) {
        // const ArrayInfo *a = &g_arrays[i];
        // printf("Array '%s'  (base=0x%llx, size=%zu bytes ~ %.3f MiB)\n",
               // a->name, (unsigned long long)a->base,
               // a->size, bytes_to_mib(a->size));
        // printf("  UM HtoD bytes:         %12llu (%.3f MiB)\n",
               // a->um_bytes_htod, bytes_to_mib(a->um_bytes_htod));
        // printf("  UM DtoH bytes:         %12llu (%.3f MiB)\n",
               // a->um_bytes_dtoh, bytes_to_mib(a->um_bytes_dtoh));
        // printf("  UM DtOD bytes:         %12llu (%.3f MiB)\n",
               // a->um_bytes_dtod, bytes_to_mib(a->um_bytes_dtod));
        // printf("  UM CPU page faults:    %12llu\n",
               // a->um_cpu_page_faults);
        // printf("  UM GPU page faults:    %12llu\n",
               // a->um_gpu_page_faults);
        // printf("------------------------------------------------------------\n");
    // }
    // printf("============================================================\n\n");
// }

static void print_pc_and_global_access_summary()
{
    printf("============================================================\n");
    printf("CUPTI Report: PC Sampling + Global Access Activity\n");
    printf("------------------------------------------------------------\n");
    if (!g_pc_sampling_enabled) {
        printf("  PC sampling:      NOT ENABLED (not compatible on this device/config)\n");
    } else {
        printf("  PC sampling records:          %12llu\n", g_pc_sampling_records);
    }

    if (!g_global_access_enabled) {
        printf("  Global access:    NOT ENABLED (not compatible on this device/config)\n");
    } else {
        printf("  Global access records:        %12llu\n", g_global_access_records);
    }

    printf("  (This demo only counts records; for source-level analysis,\n");
    printf("   build on NVIDIA's pc_sampling / sass_source_map samples.)\n");
    printf("============================================================\n\n");
}

// ======================================================================
// Main
// ======================================================================
int main(int argc, char **argv)
{
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int K = (argc > 2) ? atoi(argv[2]) : 2048;
    int N = (argc > 3) ? atoi(argv[3]) : 2048;

    printf("Running matmul with Unified Memory + CUPTI profiling\n");
    printf("  C(MxN) = A(MxK) * B(KxN)\n");
    printf("  M = %d, K = %d, N = %d\n\n", M, K, N);

    size_t sizeA = (size_t)M * (size_t)K * sizeof(float);
    size_t sizeB = (size_t)K * (size_t)N * sizeof(float);
    size_t sizeC = (size_t)M * (size_t)N * sizeof(float);

    float *A = NULL;
    float *B = NULL;
    float *C = NULL;

    // Initialize runtime & create a context (needed for CUPTI PC sampling)
    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaFree(0)); // force context creation

    // Start CUPTI
    cupti_start();

    // Use Unified Memory
    CUDA_CALL(cudaMallocManaged(&A, sizeA));
    CUDA_CALL(cudaMallocManaged(&B, sizeB));
    CUDA_CALL(cudaMallocManaged(&C, sizeC));

    // Register arrays for per-array UM stats
    register_array("A", A, sizeA);
    register_array("B", B, sizeB);
    register_array("C", C, sizeC);
    // "Other" bucket (anything that doesn't map to A/B/C)
    register_array("Other", (void*)0, (size_t)0);

    // Initialize on host
    for (size_t i = 0; i < (size_t)M * (size_t)K; ++i) A[i] = 1.0f;
    for (size_t i = 0; i < (size_t)K * (size_t)N; ++i) B[i] = 1.0f;
    for (size_t i = 0; i < (size_t)M * (size_t)N; ++i) C[i] = 0.0f;

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y);

    matmul<<<blocks, threads>>>(A, B, C, M, N, K);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    // Touch result on host
    double checksum = 0.0;
    for (size_t i = 0; i < (size_t)M * (size_t)N; ++i) checksum += C[i];

    // Stop CUPTI & print stats
    cupti_stop_and_flush();

    //print_memcpy_summary();
    print_unified_memory_summary();
    //print_array_unified_memory_summary();
    print_pc_and_global_access_summary();

    int maxRowsToPrint = (M < 4) ? M : 4;
    int maxColsToPrint = (N < 4) ? N : 4;

    printf("Top-left corner of C (up to 4x4):\n");
    for (int i = 0; i < maxRowsToPrint; ++i) {
        for (int j = 0; j < maxColsToPrint; ++j) {
            printf("%8.2f ", C[(size_t)i * (size_t)N + j]);
        }
        if (maxColsToPrint < N) printf(" ...");
        printf("\n");
    }
    if (maxRowsToPrint < M) printf("...\n");

    printf("\nResult checksum: %.4f\n", checksum);

    CUDA_CALL(cudaFree(A));
    CUDA_CALL(cudaFree(B));
    CUDA_CALL(cudaFree(C));

    return 0;
}
