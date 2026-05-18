// matmul_cupti_runtime_objects.cu
// Adds "object-aware" logging: device pointer + size, memcpy endpoints classification.
// This enables building a real buffer/object graph in Python.
//
// NOTE: This uses CUPTI runtime callback parameter structs from cupti_callbacks.h.
// If you get build errors about *_params types, see the build section below.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_callbacks.h>   // <-- gives cudaMalloc_v*_params, cudaMemcpy_v*_params, etc.
#include <nvToolsExt.h>

#define CHECK_CUDA(call) do {                                     \
  cudaError_t _e = (call);                                        \
  if (_e != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
            __FILE__, __LINE__, cudaGetErrorString(_e));          \
    std::exit(1);                                                 \
  }                                                               \
} while (0)

#define CHECK_CUPTI(call) do {                                    \
  CUptiResult _e = (call);                                        \
  if (_e != CUPTI_SUCCESS) {                                      \
    const char *errstr = nullptr;                                 \
    cuptiGetResultString(_e, &errstr);                             \
    fprintf(stderr, "CUPTI error %s:%d: %s\n",                     \
            __FILE__, __LINE__, errstr ? errstr : "unknown");     \
    std::exit(1);                                                 \
  }                                                               \
} while (0)

static constexpr size_t kActivityBufferSize = 1 << 20; // 1 MiB

static std::mutex gPrintMutex;

// correlationId -> CPU enter timestamp
static std::mutex gCorrMutex;
static std::unordered_map<uint32_t, uint64_t> gCorrStartNs;

// Device allocation table: dev_ptr -> size
static std::mutex gAllocMutex;
static std::unordered_map<uint64_t, size_t> gDevAllocs;

static std::atomic<bool> gStopFlusher{false};

static inline uint64_t now_ns() {
  using namespace std::chrono;
  return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static uint64_t ptr_u64(const void* p) {
  return (uint64_t)(uintptr_t)p;
}

// Very lightweight obfuscation (keeps joins stable but avoids printing raw addresses if you prefer).
// If you WANT raw addresses, just return ptr_u64(p).
static uint64_t ptr_id(const void* p) {
  uint64_t x = ptr_u64(p);
  // simple reversible-ish mix (not cryptographic)
  x ^= 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= (x >> 31);
  return x;
}

static const char* memcpyKindToStr(cudaMemcpyKind k) {
  switch (k) {
    case cudaMemcpyHostToDevice: return "HtoD";
    case cudaMemcpyDeviceToHost: return "DtoH";
    case cudaMemcpyDeviceToDevice: return "DtoD";
    case cudaMemcpyHostToHost: return "HtoH";
    default: return "Other";
  }
}

static void log_meminfo(const char* tag) {
  size_t freeB=0, totalB=0;
  cudaError_t e = cudaMemGetInfo(&freeB, &totalB);
  if (e != cudaSuccess) return;
  std::lock_guard<std::mutex> lk(gPrintMutex);
  printf("[MEMINFO][CPU] %s free=%zuMB total=%zuMB used=%zuMB\n",
         tag,
         freeB / (1024*1024),
         totalB / (1024*1024),
         (totalB - freeB) / (1024*1024));
}

// -------------------- CUPTI ACTIVITY (GPU timeline) --------------------

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  (void)maxNumRecords;
  uint8_t *b = (uint8_t*)std::aligned_alloc(8, kActivityBufferSize);
  if (!b) { *buffer = nullptr; *size = 0; return; }
  std::memset(b, 0, kActivityBufferSize);
  *buffer = b;
  *size = kActivityBufferSize;
}

static void printActivityRecord(const CUpti_Activity *record) {
  switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      const CUpti_ActivityMemcpy *m = (const CUpti_ActivityMemcpy*)record;
      double dur_us = (double)(m->end - m->start) / 1000.0;
      const char* k = "Other";
      switch (m->copyKind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: k = "HtoD"; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: k = "DtoH"; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: k = "DtoD"; break;
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: k = "HtoH"; break;
        default: break;
      }
      printf("[ACT][MEMCPY][GPU] %s bytes=%llu start=%llu end=%llu dur=%.3fus stream=%u corr=%u\n",
             k,
             (unsigned long long)m->bytes,
             (unsigned long long)m->start,
             (unsigned long long)m->end,
             dur_us,
             m->streamId,
             m->correlationId);
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL: {
      // Kernel4 is commonly available; if you need other versions, add guards.
      const CUpti_ActivityKernel4 *k = (const CUpti_ActivityKernel4*)record;
      double dur_us = (double)(k->end - k->start) / 1000.0;
      printf("[ACT][KERNEL][GPU] %s start=%llu end=%llu dur=%.3fus grid=(%u,%u,%u) block=(%u,%u,%u) stream=%u corr=%u\n",
             k->name ? k->name : "(null)",
             (unsigned long long)k->start,
             (unsigned long long)k->end,
             dur_us,
             k->gridX, k->gridY, k->gridZ,
             k->blockX, k->blockY, k->blockZ,
             k->streamId,
             k->correlationId);
      break;
    }
    default:
      break;
  }
}

static void CUPTIAPI bufferCompleted(CUcontext, uint32_t,
                                     uint8_t *buffer, size_t, size_t validSize) {
  CUpti_Activity *record = nullptr;
  CUptiResult status;

  std::lock_guard<std::mutex> lk(gPrintMutex);
  while ((status = cuptiActivityGetNextRecord(buffer, validSize, &record)) == CUPTI_SUCCESS) {
    printActivityRecord(record);
  }
  if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) {
    const char *errstr = nullptr;
    cuptiGetResultString(status, &errstr);
    fprintf(stderr, "cuptiActivityGetNextRecord error: %s\n", errstr ? errstr : "unknown");
  }

  size_t dropped = 0;
  cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
  if (dropped) {
    fprintf(stderr, "WARNING: dropped CUPTI activity records: %zu\n", dropped);
  }
  std::free(buffer);
}

// -------------------- CUPTI CALLBACK (CPU immediate logging + object info) --------------------

static bool is_known_dev_ptr(uint64_t dev_id, size_t* out_size) {
  std::lock_guard<std::mutex> lk(gAllocMutex);
  auto it = gDevAllocs.find(dev_id);
  if (it == gDevAllocs.end()) return false;
  if (out_size) *out_size = it->second;
  return true;
}

static void record_dev_alloc(uint64_t dev_id, size_t sz) {
  std::lock_guard<std::mutex> lk(gAllocMutex);
  gDevAllocs[dev_id] = sz;
}

static void erase_dev_alloc(uint64_t dev_id) {
  std::lock_guard<std::mutex> lk(gAllocMutex);
  gDevAllocs.erase(dev_id);
}

static void CUPTIAPI callbackHandler(void*, CUpti_CallbackDomain domain,
                                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || cbInfo == nullptr) return;

  const char* fname = cbInfo->functionName ? cbInfo->functionName : "(unknown)";
  uint64_t t = now_ns();

  // store enter time for duration
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    std::lock_guard<std::mutex> lk(gCorrMutex);
    gCorrStartNs[cbInfo->correlationId] = t;
  }

  // --- Print basic ENTER/EXIT lines (always) ---
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    std::lock_guard<std::mutex> lk(gPrintMutex);
    printf("[CB][ENTER][CPU] t=%llu ns corr=%u %s",
           (unsigned long long)t, cbInfo->correlationId, fname);

    // For memcpy, print endpoints immediately on ENTER (args are available).
    if (std::strstr(fname, "cudaMemcpy") == fname) {
      // Many CUDA versions map cudaMemcpy params to cudaMemcpy_v*_params
      // We'll try common ones.
      auto *p = (cudaMemcpy_v3020_params*)cbInfo->functionParams;
      if (p) {
        uint64_t dst_id = ptr_id(p->dst);
        uint64_t src_id = ptr_id(p->src);
        size_t dst_sz=0, src_sz=0;
        bool dst_is_dev = is_known_dev_ptr(dst_id, &dst_sz);
        bool src_is_dev = is_known_dev_ptr(src_id, &src_sz);
        printf(" dst=0x%llx src=0x%llx bytes=%zu kind=%s dst_is_dev=%d src_is_dev=%d",
               (unsigned long long)dst_id,
               (unsigned long long)src_id,
               (size_t)p->count,
               memcpyKindToStr((cudaMemcpyKind)p->kind),
               (int)dst_is_dev, (int)src_is_dev);
      }
    }

    // For cudaFree, print the pointer on ENTER.
    if (std::strcmp(fname, "cudaFree") == 0) {
      auto *p = (cudaFree_v3020_params*)cbInfo->functionParams;
      if (p) {
        uint64_t dev_id = ptr_id(p->devPtr);
        size_t known_sz=0;
        bool known = is_known_dev_ptr(dev_id, &known_sz);
        printf(" dev=0x%llx known=%d known_size=%zu",
               (unsigned long long)dev_id, (int)known, known ? known_sz : 0);
      }
    }

    printf("\n");
    return;
  }

  // EXIT site:
  uint64_t t0 = 0;
  {
    std::lock_guard<std::mutex> lk(gCorrMutex);
    auto it = gCorrStartNs.find(cbInfo->correlationId);
    if (it != gCorrStartNs.end()) { t0 = it->second; gCorrStartNs.erase(it); }
  }
  double dur_us = t0 ? (double)(t - t0) / 1000.0 : -1.0;

  // For cudaMalloc, we want the *returned* pointer value (available on EXIT).
  if (std::strcmp(fname, "cudaMalloc") == 0) {
    auto *p = (cudaMalloc_v3020_params*)cbInfo->functionParams;
    if (p && p->devPtr) {
      void* dev = *(void**)p->devPtr;   // pointer returned by cudaMalloc
      uint64_t dev_id = ptr_id(dev);
      size_t sz = (size_t)p->size;
      record_dev_alloc(dev_id, sz);

      std::lock_guard<std::mutex> lk(gPrintMutex);
      printf("[CB][EXIT ][CPU] t=%llu ns corr=%u %s  dur=%.3fus dev=0x%llx size=%zu\n",
             (unsigned long long)t, cbInfo->correlationId, fname, dur_us,
             (unsigned long long)dev_id, sz);
      return;
    }
  }

  // For cudaFree, update allocation table on EXIT (best-effort).
  if (std::strcmp(fname, "cudaFree") == 0) {
    auto *p = (cudaFree_v3020_params*)cbInfo->functionParams;
    if (p) {
      uint64_t dev_id = ptr_id(p->devPtr);
      erase_dev_alloc(dev_id);

      std::lock_guard<std::mutex> lk(gPrintMutex);
      printf("[CB][EXIT ][CPU] t=%llu ns corr=%u %s  dur=%.3fus dev=0x%llx\n",
             (unsigned long long)t, cbInfo->correlationId, fname, dur_us,
             (unsigned long long)dev_id);
      return;
    }
  }

  // Default EXIT print for everything else
  {
    std::lock_guard<std::mutex> lk(gPrintMutex);
    if (dur_us >= 0.0) {
      printf("[CB][EXIT ][CPU] t=%llu ns corr=%u %s  dur=%.3fus\n",
             (unsigned long long)t, cbInfo->correlationId, fname, dur_us);
    } else {
      printf("[CB][EXIT ][CPU] t=%llu ns corr=%u %s\n",
             (unsigned long long)t, cbInfo->correlationId, fname);
    }
  }
}

static void flusherThread(int period_ms) {
  while (!gStopFlusher.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
    cuptiActivityFlushAll(0);
  }
}

// -------------------- CUDA KERNEL --------------------

__global__ void matmulKernel(const float *A, const float *B, float *C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.0f;
    for (int i = 0; i < K; i++) sum += A[row * K + i] * B[i * N + col];
    C[row * N + col] = sum;
  }
}

// -------------------- INIT --------------------

static void initCUPTI() {
  CHECK_CUPTI(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

  CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

  CUpti_SubscriberHandle subscriber;
  CHECK_CUPTI(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callbackHandler, nullptr));
  CHECK_CUPTI(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
}

int main() {
  initCUPTI();
  std::thread flusher(flusherThread, 100);

  const int M = 512, N = 512, K = 512;
  const int iters = 6;

  const size_t bytesA = (size_t)M * K * sizeof(float);
  const size_t bytesB = (size_t)K * N * sizeof(float);
  const size_t bytesC = (size_t)M * N * sizeof(float);

  nvtxRangePushA("Host alloc/init");
  std::vector<float> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N, 0.0f);
  for (size_t i = 0; i < hA.size(); i++) hA[i] = (float)(i % 101) * 0.01f;
  for (size_t i = 0; i < hB.size(); i++) hB[i] = (float)(i % 97)  * 0.01f;
  nvtxRangePop();

  float *dA=nullptr, *dB=nullptr, *dC=nullptr;

  log_meminfo("before cudaMalloc");
  nvtxRangePushA("cudaMalloc");
  CHECK_CUDA(cudaMalloc(&dA, bytesA));
  CHECK_CUDA(cudaMalloc(&dB, bytesB));
  CHECK_CUDA(cudaMalloc(&dC, bytesC));
  nvtxRangePop();
  log_meminfo("after cudaMalloc");

  nvtxRangePushA("Initial HtoD");
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
  nvtxRangePop();

  dim3 block(16,16);
  dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

  for (int i = 0; i < iters; i++) {
    nvtxRangePushA("Iter");

    if (i % 2 == 1) {
      nvtxRangePushA("HtoD update A");
      CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
      nvtxRangePop();
    }

    nvtxRangePushA("Kernel matmul");
    matmulKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    nvtxRangePop();

    if (i % 3 == 0) {
      nvtxRangePushA("DtoH sample C");
      CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
      nvtxRangePop();
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    log_meminfo("during loop (post sync)");
    nvtxRangePop();
  }

  nvtxRangePushA("Cleanup");
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  nvtxRangePop();
  log_meminfo("after cudaFree");

  gStopFlusher.store(true);
  flusher.join();
  CHECK_CUPTI(cuptiActivityFlushAll(0));

  printf("C[0]=%.4f  C[last]=%.4f\n", hC[0], hC.back());
  return 0;
}
