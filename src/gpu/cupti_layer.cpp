/*
cupti_layer.cpp
---------------
GPU profiler for CUDA 13.2. Implements G1–G4:

  G1  Subscribe to CUPTI activity callbacks for memcpy, unified memory
      page faults, and kernel execution.

  G2  Map CUPTI activity records to GpuEvent binary format matching the
      Phase 0 schema and bridge.py wire protocol.

  G3  Timestamp synchronization — align CUPTI GPU timestamps (GPU
      hardware counter, nanoseconds from an arbitrary epoch) to
      wall-clock CLOCK_MONOTONIC nanoseconds so they line up with
      CpuEvents from the Python profiler.

  G4  Write serialized GpuEvents into a POSIX shared memory ring buffer
      (same ring_buffer.so from Phase 1) for the correlator to read.

CUDA 12.0 struct versions used:
  CUpti_ActivityMemcpy5       — memcpy (replaced Memcpy4 in CUDA 12.x)
  CUpti_ActivityKernel9       — kernel (introduced in CUDA 12.0)
  CUpti_ActivityUnifiedMemoryCounter2 — UM page faults

Build (run from project root):
  python3 src/gpu/build_gpu.py

Or manually:
  g++ -O2 -std=c++17 -shared -fPIC \
      -I/usr/local/cuda/include \
      -I/usr/local/cuda/extras/CUPTI/include \
      src/gpu/cupti_layer.cpp \
      src/cpp/ring_buffer.so \
      -L/usr/local/cuda/lib64 \
      -L/usr/local/cuda/extras/CUPTI/lib64 \
      -lcuda -lcupti \
      -Wl,-rpath,/usr/local/cuda/extras/CUPTI/lib64 \
      -o src/gpu/cupti_layer.so
*/

#include <cupti.h>
#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Ring buffer C API (from ring_buffer.so built in Phase 1)
// ---------------------------------------------------------------------------
extern "C" {
    typedef struct RingBuffer RingBuffer;
    RingBuffer* rb_create(const char* name, uint32_t capacity, int create);
    int         rb_write(RingBuffer* rb, const uint8_t* data, uint32_t len);
    void        rb_destroy(RingBuffer* rb);
    void        rb_unlink(const char* name);
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

#define CUPTI_CHECK(call)                                               \
    do {                                                                \
        CUptiResult _s = (call);                                        \
        if (_s != CUPTI_SUCCESS) {                                      \
            const char* _e = nullptr;                                   \
            cuptiGetResultString(_s, &_e);                              \
            fprintf(stderr, "[CUPTI] %s:%d  %s\n",                     \
                    __FILE__, __LINE__, _e ? _e : "unknown");           \
        }                                                               \
    } while (0)

// ---------------------------------------------------------------------------
// GpuEvent binary wire format
// Must match bridge.py _FIXED_FMT: <QQIQBBQQIbH  + GPU extension fields
//
// Base fields (matching CPU bridge format for correlator symmetry):
//   event_id         uint64
//   timestamp_ns     uint64   wall-clock ns (after G3 sync)
//   process_id       uint32
//   thread_id        uint64   (0 for GPU events)
//   event_type       uint8    2=TRANSFER 3=PAGE_FAULT 4=KERNEL
//   is_dealloc       uint8    always 0
//   alloc_address    uint64   0 (unused for GPU)
//   alloc_size_bytes uint64   0 (unused for GPU)
//   ref_count        uint32   0 (unused)
//   gc_generation    int8     0 (unused)
//   object_type_len  uint16
//   object_type      bytes    kernel name or transfer kind string
//
// GPU extension (appended immediately after object_type):
//   device_id        uint32
//   src_address      uint64
//   dst_address      uint64
//   transfer_size    uint64
//   transfer_kind    uint8    0=H2D 1=D2H 2=D2D
//   um_page_faults   uint32
// ---------------------------------------------------------------------------

static constexpr uint8_t EVENT_TRANSFER   = 2;
static constexpr uint8_t EVENT_PAGE_FAULT = 3;
static constexpr uint8_t EVENT_KERNEL     = 4;

static constexpr uint8_t XFER_H2D = 0;
static constexpr uint8_t XFER_D2H = 1;
static constexpr uint8_t XFER_D2D = 2;

static std::atomic<uint64_t> g_event_id{1};

// ---------------------------------------------------------------------------
// G3: Timestamp sync — GPU CUPTI ns → wall-clock CLOCK_MONOTONIC ns
// ---------------------------------------------------------------------------

struct TimestampSync {
    int64_t offset_ns{0};

    void calibrate() {
        // Sample 5 times and take the median offset to reduce noise.
        int64_t samples[5];
        for (int i = 0; i < 5; ++i) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            int64_t wall = (int64_t)ts.tv_sec * 1'000'000'000LL + ts.tv_nsec;
            uint64_t gpu = 0;
            cuptiGetTimestamp(&gpu);
            samples[i] = wall - (int64_t)gpu;
        }
        // Simple median of 5
        std::sort(samples, samples + 5);
        offset_ns = samples[2];
    }

    uint64_t to_wall_ns(uint64_t gpu_ts) const {
        int64_t result = (int64_t)gpu_ts + offset_ns;
        return result < 0 ? 0 : (uint64_t)result;
    }
};

// ---------------------------------------------------------------------------
// G2: Serialization helpers
// ---------------------------------------------------------------------------

static void push_u8 (std::vector<uint8_t>& b, uint8_t  v) { b.push_back(v); }
static void push_i8 (std::vector<uint8_t>& b, int8_t   v) { b.push_back((uint8_t)v); }

static void push_u16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(v & 0xFF); b.push_back((v >> 8) & 0xFF);
}
static void push_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8*i)) & 0xFF);
}
static void push_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((v >> (8*i)) & 0xFF);
}
static void push_str(std::vector<uint8_t>& b, const char* s) {
    uint16_t len = s ? (uint16_t)strnlen(s, 65535) : 0;
    push_u16(b, len);
    for (uint16_t i = 0; i < len; ++i) b.push_back((uint8_t)s[i]);
}

static std::vector<uint8_t> make_event(
    uint64_t    ts_ns,
    uint32_t    pid,
    uint8_t     event_type,
    const char* label,
    uint32_t    device_id,
    uint64_t    src_addr,
    uint64_t    dst_addr,
    uint64_t    xfer_size,
    uint8_t     xfer_kind,
    uint32_t    page_faults)
{
    std::vector<uint8_t> b;
    b.reserve(96);

    // Base fields (bridge.py _FIXED_FMT order)
    push_u64(b, g_event_id.fetch_add(1, std::memory_order_relaxed));
    push_u64(b, ts_ns);
    push_u32(b, pid);
    push_u64(b, 0);          // thread_id
    push_u8 (b, event_type);
    push_u8 (b, 0);          // is_dealloc
    push_u64(b, 0);          // alloc_address
    push_u64(b, 0);          // alloc_size_bytes
    push_u32(b, 0);          // ref_count
    push_i8 (b, 0);          // gc_generation
    push_str(b, label);      // object_type_len + object_type

    // GPU extension fields
    push_u32(b, device_id);
    push_u64(b, src_addr);
    push_u64(b, dst_addr);
    push_u64(b, xfer_size);
    push_u8 (b, xfer_kind);
    push_u32(b, page_faults);

    return b;
}

// ---------------------------------------------------------------------------
// Transfer kind helpers
// ---------------------------------------------------------------------------

static uint8_t memcpy_kind(uint8_t cupti_kind) {
    switch (cupti_kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return XFER_H2D;
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return XFER_D2H;
        default:                              return XFER_D2D;
    }
}

static const char* memcpy_label(uint8_t cupti_kind) {
    switch (cupti_kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HOST_TO_DEVICE";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DEVICE_TO_HOST";
        default:                              return "DEVICE_TO_DEVICE";
    }
}

// ---------------------------------------------------------------------------
// G1+G4: CuptiLayer — manages CUPTI subscription and ring buffer writes
// ---------------------------------------------------------------------------

class CuptiLayer {
public:
    // Singleton — CUPTI callbacks have no user-data parameter
    static CuptiLayer* instance;

    CuptiLayer() = default;
    ~CuptiLayer() { stop(); }

    // -- Public API ----------------------------------------------------------

    bool start(const char* shm_name, uint32_t capacity = 8 * 1024 * 1024) {
        if (_running) return true;

        _rb = rb_create(shm_name, capacity, /*create=*/1);
        if (!_rb) {
            fprintf(stderr, "[CuptiLayer] Failed to create shm '%s'\n", shm_name);
            return false;
        }

        _pid = (uint32_t)getpid();
        _sync.calibrate();
        instance = this;

        // G1: register buffer callbacks
        CUPTI_CHECK(cuptiActivityRegisterCallbacks(
            _buf_request, _buf_complete));

        // G1: enable activity kinds
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

        _running = true;
        fprintf(stderr, "[CuptiLayer] started, shm=%s\n", shm_name);
        return true;
    }

    void stop() {
        if (!_running) return;

        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

        // Flush any pending activity records
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

        if (_rb) { rb_destroy(_rb); _rb = nullptr; }
        instance  = nullptr;
        _running  = false;

        fprintf(stderr, "[CuptiLayer] stopped, events written=%llu\n",
                (unsigned long long)_written.load());
    }

    uint64_t written() const { return _written.load(); }

    // -- CUPTI buffer callbacks (static, routed through instance) ------------

    static void CUPTIAPI _buf_request(
        uint8_t** buffer, size_t* size, size_t* maxNumRecords)
    {
        constexpr size_t BUF_SIZE  = 8 * 1024 * 1024;
        constexpr size_t ALIGN     = 8;
        uint8_t* b = nullptr;
        if (posix_memalign((void**)&b, ALIGN, BUF_SIZE) != 0) {
            *buffer = nullptr; *size = 0; *maxNumRecords = 0;
            return;
        }
        *buffer        = b;
        *size          = BUF_SIZE;
        *maxNumRecords = 0;  // fill buffer completely
    }

    static void CUPTIAPI _buf_complete(
        CUcontext ctx, uint32_t streamId,
        uint8_t* buffer, size_t /*size*/, size_t validSize)
    {
        if (instance) instance->_drain(ctx, streamId, buffer, validSize);
        else          free(buffer);
    }

private:

    // -- G2: drain one CUPTI buffer and write GpuEvents to ring buffer -------

    void _drain(CUcontext ctx, uint32_t streamId,
                uint8_t* buffer, size_t validSize)
    {
        if (!_rb || validSize == 0) { free(buffer); return; }

        CUpti_Activity* rec = nullptr;
        CUptiResult status;

        while (true) {
            status = cuptiActivityGetNextRecord(buffer, validSize, &rec);
            if (status == CUPTI_SUCCESS) {
                _process(rec);
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                break;
            }
        }

        // Log dropped records
        size_t dropped = 0;
        cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
        if (dropped > 0)
            fprintf(stderr, "[CuptiLayer] %zu records dropped\n", dropped);

        free(buffer);
    }

    void _process(CUpti_Activity* rec) {
        switch (rec->kind) {

        // G2: memcpy record (CUpti_ActivityMemcpy5 — CUDA 12.x+)
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto* r = reinterpret_cast<CUpti_ActivityMemcpy5*>(rec);
            auto buf = make_event(
                _sync.to_wall_ns(r->start),
                _pid,
                EVENT_TRANSFER,
                memcpy_label(r->copyKind),
                r->deviceId,
                0,           // src host address not exposed by CUPTI
                0,           // dst host address not exposed by CUPTI
                r->bytes,
                memcpy_kind(r->copyKind),
                0            // no page faults for explicit memcpy
            );
            _write(buf);
            break;
        }

        // G2: unified memory counter (page faults)
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: {
            auto* r = reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2*>(rec);

            // Only capture GPU and CPU page fault records
            if (r->counterKind !=
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT &&
                r->counterKind !=
                    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT)
                break;

            auto buf = make_event(
                _sync.to_wall_ns(r->end),
                _pid,
                EVENT_PAGE_FAULT,
                "page_fault",
                r->dstId,
                (uint64_t)r->address,   // faulting virtual address
                0,
                0,
                XFER_D2D,
                (uint32_t)r->value      // fault count
            );
            _write(buf);
            break;
        }

        // G2: kernel record (CUpti_ActivityKernel9 — CUDA 12.0)
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto* r = reinterpret_cast<CUpti_ActivityKernel9*>(rec);
            const char* name = r->name ? r->name : "unknown_kernel";
            auto buf = make_event(
                _sync.to_wall_ns(r->start),
                _pid,
                EVENT_KERNEL,
                name,
                r->deviceId,
                0, 0, 0,    // no addresses for kernel records
                XFER_D2D,
                0
            );
            _write(buf);
            break;
        }

        default: break;
        }
    }

    // G4: write one event to the ring buffer
    void _write(const std::vector<uint8_t>& buf) {
        if (rb_write(_rb, buf.data(), (uint32_t)buf.size()))
            _written.fetch_add(1, std::memory_order_relaxed);
    }

    RingBuffer*           _rb      = nullptr;
    uint32_t              _pid     = 0;
    bool                  _running = false;
    TimestampSync         _sync;
    std::atomic<uint64_t> _written{0};
};

CuptiLayer* CuptiLayer::instance = nullptr;

// ---------------------------------------------------------------------------
// C API — callable from Python (ctypes) or as a standalone profiler binary
// ---------------------------------------------------------------------------

extern "C" {

static CuptiLayer* g_layer = nullptr;

/*
 * cupti_start(shm_name)
 * Open the GPU ring buffer and start CUPTI activity collection.
 * Returns 1 on success, 0 on failure.
 */
int cupti_start(const char* shm_name) {
    if (g_layer) return 0;
    g_layer = new CuptiLayer();
    if (!g_layer->start(shm_name)) {
        delete g_layer;
        g_layer = nullptr;
        return 0;
    }
    return 1;
}

/*
 * cupti_stop()
 * Flush remaining records, close the ring buffer, destroy the layer.
 */
void cupti_stop() {
    if (!g_layer) return;
    g_layer->stop();
    delete g_layer;
    g_layer = nullptr;
}

/*
 * cupti_events_written()
 * Returns the number of GpuEvents written to the ring buffer so far.
 */
uint64_t cupti_events_written() {
    return g_layer ? g_layer->written() : 0;
}

} // extern "C"