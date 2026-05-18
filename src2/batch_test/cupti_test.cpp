// cupti_trace.cpp  (CUDA 12.0 friendly; best-effort arg decoding)
// Build: g++ -shared -fPIC ...

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <thread>

#include <cuda_runtime.h>
#include <cupti.h>
#include <cupti_callbacks.h>

static std::mutex gPrint;
static std::mutex gCorr;
static std::unordered_map<uint32_t, uint64_t> gCorrStartNs;

static std::atomic<bool> gStop{false};
static FILE* gOut = nullptr;

static inline uint64_t now_ns() {
  using namespace std::chrono;
  return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

static void out_printf(const char* fmt, ...) {
  std::lock_guard<std::mutex> lk(gPrint);
  if (!gOut) gOut = stdout;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(gOut, fmt, ap);
  va_end(ap);
  fputc('\n', gOut);
  fflush(gOut);
}

static const char* cupti_errstr(CUptiResult r) {
  const char* s = nullptr;
  cuptiGetResultString(r, &s);
  return s ? s : "unknown";
}

#define CUPTI_OK(call) do { \
  CUptiResult _r = (call); \
  if (_r != CUPTI_SUCCESS) { \
    out_printf("{\"type\":\"error\",\"where\":\"cupti\",\"msg\":\"%s\",\"code\":%d}", cupti_errstr(_r), (int)_r); \
  } \
} while(0)

static const char* memcpyKindToStr(cudaMemcpyKind k) {
  switch(k) {
    case cudaMemcpyHostToDevice: return "HtoD";
    case cudaMemcpyDeviceToHost: return "DtoH";
    case cudaMemcpyDeviceToDevice: return "DtoD";
    case cudaMemcpyHostToHost: return "HtoH";
    default: return "Other";
  }
}

// ---------------- Activity API (GPU truth) ----------------
static constexpr size_t kBufSize = 1 << 20;

static void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  (void)maxNumRecords;
  uint8_t* b = (uint8_t*)std::aligned_alloc(8, kBufSize);
  if (!b) { *buffer=nullptr; *size=0; return; }
  std::memset(b, 0, kBufSize);
  *buffer = b;
  *size = kBufSize;
}

static void printActivity(const CUpti_Activity* rec) {
  if (!rec) return;

  if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
    auto* m = (const CUpti_ActivityMemcpy*)rec;
    double dur_us = (double)(m->end - m->start) / 1000.0;

    const char* op = "Other";
    switch (m->copyKind) {
      case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: op = "HtoD"; break;
      case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: op = "DtoH"; break;
      case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: op = "DtoD"; break;
      case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: op = "HtoH"; break;
      default: break;
    }

    out_printf(
      "{\"type\":\"activity\",\"side\":\"gpu\",\"kind\":\"memcpy\",\"op\":\"%s\",\"bytes\":%llu,"
      "\"start\":%llu,\"end\":%llu,\"dur_us\":%.3f,\"stream\":%u,\"corr\":%u}",
      op, (unsigned long long)m->bytes,
      (unsigned long long)m->start, (unsigned long long)m->end,
      dur_us, m->streamId, m->correlationId
    );
  }

  if (rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
    // Kernel4 is widely available; if your install uses a different version, adjust here.
    auto* k = (const CUpti_ActivityKernel4*)rec;
    double dur_us = (double)(k->end - k->start) / 1000.0;

    out_printf(
      "{\"type\":\"activity\",\"side\":\"gpu\",\"kind\":\"kernel\",\"name\":\"%s\","
      "\"start\":%llu,\"end\":%llu,\"dur_us\":%.3f,"
      "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"stream\":%u,\"corr\":%u}",
      k->name ? k->name : "(null)",
      (unsigned long long)k->start, (unsigned long long)k->end,
      dur_us,
      k->gridX, k->gridY, k->gridZ,
      k->blockX, k->blockY, k->blockZ,
      k->streamId, k->correlationId
    );
  }
}

static void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  CUpti_Activity* rec = nullptr;
  CUptiResult st;
  while ((st = cuptiActivityGetNextRecord(buffer, validSize, &rec)) == CUPTI_SUCCESS) {
    printActivity(rec);
  }
  std::free(buffer);
}

static void flusher_thread() {
  while (!gStop.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    cuptiActivityFlushAll(0);
  }
}

// ------------- Callback API (CPU orchestration) -------------
static void log_meminfo(const char* tag, uint64_t t_ns) {
  size_t freeB=0, totalB=0;
  if (cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
    out_printf(
      "{\"type\":\"meminfo\",\"side\":\"device\",\"tag\":\"%s\",\"t_ns\":%llu,"
      "\"free_mb\":%zu,\"total_mb\":%zu,\"used_mb\":%zu}",
      tag, (unsigned long long)t_ns,
      freeB/(1024*1024), totalB/(1024*1024), (totalB-freeB)/(1024*1024)
    );
  }
}

// Best-effort arg decoding. If the *_v3020_params typedefs exist, you get args; if not, you still get timing+corr.
static void CUPTIAPI callbackHandler(void*, CUpti_CallbackDomain domain, CUpti_CallbackId,
                                     const CUpti_CallbackData* cbInfo) {
  if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || !cbInfo) return;

  const char* api = cbInfo->functionName ? cbInfo->functionName : "(unknown)";
  uint32_t corr = cbInfo->correlationId;
  uint64_t t = now_ns();

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    {
      std::lock_guard<std::mutex> lk(gCorr);
      gCorrStartNs[corr] = t;
    }

    // Emit args if available
#if defined(cudaMemcpy_v3020_params)
    if (std::strcmp(api, "cudaMemcpy") == 0) {
      auto* p = (cudaMemcpy_v3020_params*)cbInfo->functionParams;
      if (p) {
        out_printf(
          "{\"type\":\"callback\",\"side\":\"cpu\",\"site\":\"enter\",\"api\":\"%s\",\"corr\":%u,\"t_ns\":%llu,"
          "\"dst\":%llu,\"src\":%llu,\"bytes\":%zu,\"kind\":\"%s\"}",
          api, corr, (unsigned long long)t,
          (unsigned long long)(uintptr_t)p->dst,
          (unsigned long long)(uintptr_t)p->src,
          (size_t)p->count,
          memcpyKindToStr((cudaMemcpyKind)p->kind)
        );
        return;
      }
    }
#endif

#if defined(cudaMalloc_v3020_params)
    if (std::strcmp(api, "cudaMalloc") == 0) {
      auto* p = (cudaMalloc_v3020_params*)cbInfo->functionParams;
      if (p) {
        out_printf(
          "{\"type\":\"callback\",\"side\":\"cpu\",\"site\":\"enter\",\"api\":\"%s\",\"corr\":%u,\"t_ns\":%llu,\"bytes\":%zu}",
          api, corr, (unsigned long long)t, (size_t)p->size
        );
        return;
      }
    }
#endif

#if defined(cudaFree_v3020_params)
    if (std::strcmp(api, "cudaFree") == 0) {
      auto* p = (cudaFree_v3020_params*)cbInfo->functionParams;
      if (p) {
        out_printf(
          "{\"type\":\"callback\",\"side\":\"cpu\",\"site\":\"enter\",\"api\":\"%s\",\"corr\":%u,\"t_ns\":%llu,\"ptr\":%llu}",
          api, corr, (unsigned long long)t, (unsigned long long)(uintptr_t)p->devPtr
        );
        return;
      }
    }
#endif

    out_printf(
      "{\"type\":\"callback\",\"side\":\"cpu\",\"site\":\"enter\",\"api\":\"%s\",\"corr\":%u,\"t_ns\":%llu}",
      api, corr, (unsigned long long)t
    );
    return;
  }

  // EXIT
  uint64_t t0 = 0;
  {
    std::lock_guard<std::mutex> lk(gCorr);
    auto it = gCorrStartNs.find(corr);
    if (it != gCorrStartNs.end()) { t0 = it->second; gCorrStartNs.erase(it); }
  }
  double dur_us = t0 ? (double)(t - t0) / 1000.0 : -1.0;

  // Device mem snapshots at key boundaries (cheap and very informative)
  if (std::strcmp(api, "cudaMalloc") == 0) log_meminfo("cudaMalloc_exit", t);
  if (std::strcmp(api, "cudaFree") == 0)   log_meminfo("cudaFree_exit", t);

  out_printf(
    "{\"type\":\"callback\",\"side\":\"cpu\",\"site\":\"exit\",\"api\":\"%s\",\"corr\":%u,\"t_ns\":%llu,\"dur_us\":%.3f}",
    api, corr, (unsigned long long)t, dur_us
  );
}

// ------------- Init / teardown -------------
static void init_tracer() {
  const char* outPath = std::getenv("CUPTI_TRACE_OUT");
  if (outPath && outPath[0]) {
    gOut = std::fopen(outPath, "w");
    if (!gOut) gOut = stdout;
  } else {
    gOut = stdout;
  }

  out_printf("{\"type\":\"meta\",\"msg\":\"cupti tracer init\"}");

  CUPTI_OK(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
  CUPTI_OK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_OK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

  CUpti_SubscriberHandle sub;
  CUPTI_OK(cuptiSubscribe(&sub, (CUpti_CallbackFunc)callbackHandler, nullptr));
  CUPTI_OK(cuptiEnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API));

  std::thread(flusher_thread).detach();
}

static void shutdown_tracer() {
  gStop.store(true);
  cuptiActivityFlushAll(0);
  out_printf("{\"type\":\"meta\",\"msg\":\"cupti tracer shutdown\"}");
  if (gOut && gOut != stdout) std::fclose(gOut);
  gOut = nullptr;
}

__attribute__((constructor)) static void on_load()  { init_tracer(); }
__attribute__((destructor))  static void on_unload(){ shutdown_tracer(); }