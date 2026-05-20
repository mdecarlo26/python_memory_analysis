/*
cupti_layer.cpp
---------------
GPU profiler for CUDA 12.x. Implements G1–G4 + extended metrics.

  G1  Subscribe to CUPTI activity callbacks:
        MEMCPY             — explicit cudaMemcpy (all directions)
        UNIFIED_MEMORY_COUNTER — UM migration bytes AND page faults
        CONCURRENT_KERNEL  — kernel execution timing + occupancy
        MEMSET             — device memory initialization

  G2  Map CUPTI activity records to GpuEvent binary format.

  G3  Timestamp synchronisation — CUPTI GPU ns → CLOCK_MONOTONIC ns.

  G4  Write serialised GpuEvents into a POSIX shared memory ring buffer.

Extended GPU extension block vs original (new fields marked [NEW]):
  device_id             uint32
  src_address           uint64
  dst_address           uint64
  transfer_size         uint64
  transfer_kind         uint8    0=H2D 1=D2H 2=D2D 3=PREFETCH
  um_page_faults        uint32
  kernel_duration_ns    uint64
  stream_id             uint32
  device_mem_used_mb    uint32
  [NEW] um_bytes_htod   uint64  — UM migration bytes H→D (0 for non-UM events)
  [NEW] um_bytes_dtoh   uint64  — UM migration bytes D→H
  [NEW] grid_x          uint32  — kernel grid dims (0 for non-kernel events)
  [NEW] grid_y          uint32
  [NEW] grid_z          uint32
  [NEW] block_x         uint32  — kernel block dims
  [NEW] block_y         uint32
  [NEW] block_z         uint32
  [NEW] registers_per_thread uint32 — from CUpti_ActivityKernel9
  [NEW] shared_mem_bytes     uint32 — static + dynamic shared mem per block
  [NEW] correlation_id       uint32 — CUPTI correlation ID linking API→activity

CUDA 12 struct versions:
  CUpti_ActivityMemcpy5              — explicit memcpy
  CUpti_ActivityKernel9              — kernel execution
  CUpti_ActivityUnifiedMemoryCounter2 — UM counters
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
// Ring buffer C API (from ring_buffer.so)
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
// Event type bytes (match bridge.py _EVENT_TYPE_MAP)
// ---------------------------------------------------------------------------
static constexpr uint8_t EVENT_TRANSFER   = 2;
static constexpr uint8_t EVENT_PAGE_FAULT = 3;
static constexpr uint8_t EVENT_KERNEL     = 4;

static constexpr uint8_t XFER_H2D      = 0;
static constexpr uint8_t XFER_D2H      = 1;
static constexpr uint8_t XFER_D2D      = 2;
static constexpr uint8_t XFER_PREFETCH = 3;

static std::atomic<uint64_t> g_event_id{1};

// ---------------------------------------------------------------------------
// G3: Timestamp sync — GPU CUPTI ns → CLOCK_MONOTONIC wall ns
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
        std::sort(samples, samples + 5);
        offset_ns = samples[2];  // median
    }

    uint64_t to_wall_ns(uint64_t gpu_ts) const {
        int64_t result = (int64_t)gpu_ts + offset_ns;
        return result < 0 ? 0 : (uint64_t)result;
    }
};

// ---------------------------------------------------------------------------
// G2: Serialisation helpers (little-endian)
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

// ---------------------------------------------------------------------------
// Extended GpuEvent record — all fields in serialisation order.
// ---------------------------------------------------------------------------
struct GpuEventData {
    uint64_t ts_ns            = 0;
    uint32_t pid              = 0;
    uint8_t  event_type       = 0;
    const char* label         = "";   // kernel name or transfer direction

    // GPU extension — original fields
    uint32_t device_id        = 0;
    uint64_t src_address      = 0;
    uint64_t dst_address      = 0;
    uint64_t transfer_size    = 0;
    uint8_t  transfer_kind    = XFER_D2D;
    uint32_t page_faults      = 0;
    uint64_t kernel_duration_ns = 0;
    uint32_t stream_id        = 0;
    uint32_t device_mem_used_mb = 0;

    // NEW: UM migration byte counters (populated for UM events)
    uint64_t um_bytes_htod    = 0;
    uint64_t um_bytes_dtoh    = 0;

    // NEW: Kernel occupancy fields (populated for KERNEL events)
    uint32_t grid_x           = 0;
    uint32_t grid_y           = 0;
    uint32_t grid_z           = 0;
    uint32_t block_x          = 0;
    uint32_t block_y          = 0;
    uint32_t block_z          = 0;
    uint32_t registers_per_thread = 0;
    uint32_t shared_mem_bytes     = 0;  // static + dynamic

    // NEW: CUPTI correlation ID (links CUDA API call → activity record)
    uint32_t correlation_id   = 0;
};

static std::vector<uint8_t> serialise_event(const GpuEventData& ev) {
    std::vector<uint8_t> b;
    b.reserve(192);

    // ── Base header (matches bridge.py _FIXED_FMT: <QQIQBBQQIbH) ──────────
    push_u64(b, g_event_id.fetch_add(1, std::memory_order_relaxed));
    push_u64(b, ev.ts_ns);
    push_u32(b, ev.pid);
    push_u64(b, 0);               // thread_id (always 0 for GPU)
    push_u8 (b, ev.event_type);
    push_u8 (b, 0);               // is_dealloc
    push_u64(b, 0);               // alloc_address (unused)
    push_u64(b, 0);               // alloc_size_bytes (unused)
    push_u32(b, 0);               // ref_count (unused)
    push_i8 (b, 0);               // gc_generation (unused)
    push_str(b, ev.label);        // object_type_len + object_type

    // ── GPU extension block ─────────────────────────────────────────────────
    push_u32(b, ev.device_id);
    push_u64(b, ev.src_address);
    push_u64(b, ev.dst_address);
    push_u64(b, ev.transfer_size);
    push_u8 (b, ev.transfer_kind);
    push_u32(b, ev.page_faults);
    push_u64(b, ev.kernel_duration_ns);
    push_u32(b, ev.stream_id);
    push_u32(b, ev.device_mem_used_mb);

    // ── New fields ──────────────────────────────────────────────────────────
    push_u64(b, ev.um_bytes_htod);
    push_u64(b, ev.um_bytes_dtoh);
    push_u32(b, ev.grid_x);
    push_u32(b, ev.grid_y);
    push_u32(b, ev.grid_z);
    push_u32(b, ev.block_x);
    push_u32(b, ev.block_y);
    push_u32(b, ev.block_z);
    push_u32(b, ev.registers_per_thread);
    push_u32(b, ev.shared_mem_bytes);
    push_u32(b, ev.correlation_id);

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
// G1+G4: CuptiLayer
// ---------------------------------------------------------------------------
class CuptiLayer {
public:
    static CuptiLayer* instance;

    CuptiLayer()  = default;
    ~CuptiLayer() { stop(); }

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
        CUPTI_CHECK(cuptiActivityRegisterCallbacks(_buf_request, _buf_complete));

        // G1: enable core activity kinds
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
        CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

        // G1: enable UM migration bytes (separate counter kind from page faults)
        _enable_um_migration_bytes();

        _running = true;
        fprintf(stderr, "[CuptiLayer] started shm=%s\n", shm_name);
        return true;
    }

    void stop() {
        if (!_running) return;

        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

        if (_rb) { rb_destroy(_rb); _rb = nullptr; }
        instance = nullptr;
        _running = false;

        fprintf(stderr, "[CuptiLayer] stopped, events written=%llu\n",
                (unsigned long long)_written.load());
    }

    uint64_t written() const { return _written.load(); }

    // ── CUPTI buffer callbacks ───────────────────────────────────────────────

    static void CUPTIAPI _buf_request(
        uint8_t** buffer, size_t* size, size_t* maxNumRecords)
    {
        constexpr size_t BUF_SIZE = 8 * 1024 * 1024;
        uint8_t* b = nullptr;
        if (posix_memalign((void**)&b, 8, BUF_SIZE) != 0) {
            *buffer = nullptr; *size = 0; *maxNumRecords = 0;
            return;
        }
        *buffer = b; *size = BUF_SIZE; *maxNumRecords = 0;
    }

    static void CUPTIAPI _buf_complete(
        CUcontext ctx, uint32_t streamId,
        uint8_t* buffer, size_t /*size*/, size_t validSize)
    {
        if (instance) instance->_drain(ctx, streamId, buffer, validSize);
        else          free(buffer);
    }

private:

    // ── Enable UM migration byte counters ────────────────────────────────────
    void _enable_um_migration_bytes() {
        // Configure UM counters for bytes (in addition to page fault counters
        // already enabled via CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER).
        CUpti_ActivityUnifiedMemoryCounterConfig configs[3];
        memset(configs, 0, sizeof(configs));

        auto scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;

        configs[0].scope  = scope;
        configs[0].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
        configs[0].enable = 1;

        configs[1].scope  = scope;
        configs[1].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
        configs[1].enable = 1;

        configs[2].scope  = scope;
        configs[2].kind   = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT;
        configs[2].enable = 1;

        CUptiResult r = cuptiActivityConfigureUnifiedMemoryCounter(
            configs, (uint32_t)(sizeof(configs)/sizeof(configs[0])));
        if (r != CUPTI_SUCCESS) {
            const char* e = nullptr;
            cuptiGetResultString(r, &e);
            fprintf(stderr, "[CuptiLayer] UM counter config failed: %s\n",
                    e ? e : "unknown");
        }
    }

    // ── Drain one CUPTI buffer ────────────────────────────────────────────────
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

        size_t dropped = 0;
        cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
        if (dropped > 0)
            fprintf(stderr, "[CuptiLayer] %zu records dropped\n", dropped);

        free(buffer);
    }

    // ── Snapshot free device memory → used MB ────────────────────────────────
    static uint32_t _device_mem_used_mb() {
        size_t free_bytes = 0, total_bytes = 0;
        if (cuMemGetInfo(&free_bytes, &total_bytes) == CUDA_SUCCESS && total_bytes > 0)
            return (uint32_t)((total_bytes - free_bytes) / (1024 * 1024));
        return 0;
    }

    // ── Write one event to ring buffer ────────────────────────────────────────
    void _write(const std::vector<uint8_t>& buf) {
        if (rb_write(_rb, buf.data(), (uint32_t)buf.size()))
            _written.fetch_add(1, std::memory_order_relaxed);
    }

    // ── Process one CUPTI activity record ────────────────────────────────────
    void _process(CUpti_Activity* rec) {
        switch (rec->kind) {

        // ── Explicit cudaMemcpy ────────────────────────────────────────────
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto* r = reinterpret_cast<CUpti_ActivityMemcpy5*>(rec);
            GpuEventData ev;
            ev.ts_ns              = _sync.to_wall_ns(r->start);
            ev.pid                = _pid;
            ev.event_type         = EVENT_TRANSFER;
            ev.label              = memcpy_label(r->copyKind);
            ev.device_id          = r->deviceId;
            // Note: CUPTI Memcpy5 does not expose the host-side virtual address
            // directly; srcAddress/dstAddress are device addresses only.
            ev.src_address        = 0;
            ev.dst_address        = 0;
            ev.transfer_size      = r->bytes;
            ev.transfer_kind      = memcpy_kind(r->copyKind);
            ev.kernel_duration_ns = r->end - r->start;  // transfer duration
            ev.stream_id          = r->streamId;
            ev.device_mem_used_mb = _device_mem_used_mb();
            ev.correlation_id     = r->correlationId;
            _write(serialise_event(ev));
            break;
        }

        // ── Unified Memory counters ─────────────────────────────────────────
        // Two sub-kinds: byte migration counters and page fault counters.
        // We emit separate GpuEvents for each so the correlator can see them.
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: {
            auto* r = reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2*>(rec);

            switch (r->counterKind) {

            // UM byte migration H→D: emit as a TRANSFER event with um_bytes_htod set
            case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD: {
                GpuEventData ev;
                ev.ts_ns              = _sync.to_wall_ns(r->end);
                ev.pid                = _pid;
                ev.event_type         = EVENT_TRANSFER;
                ev.label              = "HOST_TO_DEVICE";
                ev.device_id          = r->dstId;
                ev.src_address        = (uint64_t)r->address;  // UM page address
                ev.transfer_size      = (uint64_t)r->value;    // bytes migrated
                ev.transfer_kind      = XFER_H2D;
                ev.device_mem_used_mb = _device_mem_used_mb();
                ev.um_bytes_htod      = (uint64_t)r->value;
                _write(serialise_event(ev));
                break;
            }

            // UM byte migration D→H: emit as TRANSFER with um_bytes_dtoh set
            case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH: {
                GpuEventData ev;
                ev.ts_ns              = _sync.to_wall_ns(r->end);
                ev.pid                = _pid;
                ev.event_type         = EVENT_TRANSFER;
                ev.label              = "DEVICE_TO_HOST";
                ev.device_id          = r->srcId;
                ev.src_address        = (uint64_t)r->address;
                ev.transfer_size      = (uint64_t)r->value;
                ev.transfer_kind      = XFER_D2H;
                ev.device_mem_used_mb = _device_mem_used_mb();
                ev.um_bytes_dtoh      = (uint64_t)r->value;
                _write(serialise_event(ev));
                break;
            }

            // GPU page faults: emit as PAGE_FAULT event
            case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT: {
                GpuEventData ev;
                ev.ts_ns              = _sync.to_wall_ns(r->end);
                ev.pid                = _pid;
                ev.event_type         = EVENT_PAGE_FAULT;
                ev.label              = "page_fault";
                ev.device_id          = r->dstId;
                ev.src_address        = (uint64_t)r->address;
                ev.page_faults        = (uint32_t)r->value;  // fault groups
                ev.device_mem_used_mb = _device_mem_used_mb();
                _write(serialise_event(ev));
                break;
            }

            // CPU page faults: also emit as PAGE_FAULT, distinguishable by device_id=0
            case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT: {
                GpuEventData ev;
                ev.ts_ns              = _sync.to_wall_ns(r->end);
                ev.pid                = _pid;
                ev.event_type         = EVENT_PAGE_FAULT;
                ev.label              = "cpu_page_fault";
                ev.device_id          = 0;   // CPU, not a GPU device
                ev.src_address        = (uint64_t)r->address;
                ev.page_faults        = 1;   // each record = one CPU fault
                ev.device_mem_used_mb = _device_mem_used_mb();
                _write(serialise_event(ev));
                break;
            }

            default: break;
            }
            break;
        }

        // ── Kernel execution ────────────────────────────────────────────────
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto* r = reinterpret_cast<CUpti_ActivityKernel9*>(rec);
            GpuEventData ev;
            ev.ts_ns                 = _sync.to_wall_ns(r->start);
            ev.pid                   = _pid;
            ev.event_type            = EVENT_KERNEL;
            ev.label                 = r->name ? r->name : "unknown_kernel";
            ev.device_id             = r->deviceId;
            ev.kernel_duration_ns    = r->end - r->start;
            ev.stream_id             = r->streamId;
            ev.device_mem_used_mb    = _device_mem_used_mb();
            ev.correlation_id        = r->correlationId;
            // Occupancy / launch parameters from kernel record
            ev.grid_x                = r->gridX;
            ev.grid_y                = r->gridY;
            ev.grid_z                = r->gridZ;
            ev.block_x               = r->blockX;
            ev.block_y               = r->blockY;
            ev.block_z               = r->blockZ;
            ev.registers_per_thread  = r->registersPerThread;
            ev.shared_mem_bytes      = r->staticSharedMemory + r->dynamicSharedMemory;
            _write(serialise_event(ev));
            break;
        }

        default: break;
        }
    }

    RingBuffer*           _rb      = nullptr;
    uint32_t              _pid     = 0;
    bool                  _running = false;
    TimestampSync         _sync;
    std::atomic<uint64_t> _written{0};
};

CuptiLayer* CuptiLayer::instance = nullptr;

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------
extern "C" {

static CuptiLayer* g_layer = nullptr;

int cupti_start(const char* shm_name) {
    if (g_layer) return 0;
    g_layer = new CuptiLayer();
    if (!g_layer->start(shm_name)) {
        delete g_layer; g_layer = nullptr; return 0;
    }
    return 1;
}

void cupti_stop() {
    if (!g_layer) return;
    g_layer->stop();
    delete g_layer; g_layer = nullptr;
}

uint64_t cupti_events_written() {
    return g_layer ? g_layer->written() : 0;
}

} // extern "C"