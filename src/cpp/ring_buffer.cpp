/*
ring_buffer.cpp
---------------
Lock-free single-producer single-consumer ring buffer over a POSIX
shared memory segment. Used to transport serialized CpuEvents from
the Python profiler (writer) to the C++ correlator (reader) across
process boundaries.

Layout of the shared memory segment:
  [0..sizeof(RingBufferHeader)] — header (capacity, head, tail, flags)
  [sizeof(RingBufferHeader)..N]  — data region, capacity bytes

Each record in the data region is prefixed with a 4-byte little-endian
length, followed by that many bytes of payload (the serialized event).

  | uint32 len | <len bytes of payload> | uint32 len | ...

Designed as a C shared library loaded by Python via cffi.
Also usable directly from C++ by including this header.

Build:
  g++ -O2 -shared -fPIC -o ring_buffer.so ring_buffer.cpp \
      -lrt -std=c++17
*/

#include <cstdint>
#include <cstring>
#include <atomic>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// -------------------------------------------------------------------------
// Constants
// -------------------------------------------------------------------------

static constexpr uint32_t MAGIC         = 0x48504352; // 'HPCR'
static constexpr uint32_t VERSION       = 1;
static constexpr size_t   HEADER_SIZE   = 64;          // padded to cache line
static constexpr size_t   DEFAULT_CAP   = 4 * 1024 * 1024; // 4 MB

// -------------------------------------------------------------------------
// Shared memory header — lives at byte 0 of the mmap region
// -------------------------------------------------------------------------

struct alignas(64) RingBufferHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t capacity;      // usable data bytes (total_size - HEADER_SIZE)
    uint32_t _pad0;

    std::atomic<uint32_t> head;  // read cursor (consumer advances)
    std::atomic<uint32_t> tail;  // write cursor (producer advances)
    std::atomic<uint32_t> dropped; // events dropped due to full buffer
    uint32_t _pad1;
};

static_assert(sizeof(RingBufferHeader) <= HEADER_SIZE,
    "RingBufferHeader exceeds HEADER_SIZE");

// -------------------------------------------------------------------------
// Handle passed back to callers
// -------------------------------------------------------------------------

struct RingBuffer {
    int          fd;
    void*        mem;
    size_t       total_size;
    RingBufferHeader* hdr;
    uint8_t*     data;     // points to mem + HEADER_SIZE
};

// -------------------------------------------------------------------------
// Internal helpers
// -------------------------------------------------------------------------

static inline uint32_t rb_used(const RingBufferHeader* hdr) {
    uint32_t h = hdr->head.load(std::memory_order_acquire);
    uint32_t t = hdr->tail.load(std::memory_order_acquire);
    return (t - h) & (hdr->capacity - 1);
}

static inline uint32_t rb_free(const RingBufferHeader* hdr) {
    return hdr->capacity - rb_used(hdr) - 1;
}

// -------------------------------------------------------------------------
// Public C API (callable from Python via cffi)
// -------------------------------------------------------------------------

extern "C" {

/*
 * rb_create — create or open a named shared memory ring buffer.
 *
 * name:     POSIX shm name, e.g. "/hpc_profiler_cpu"
 * capacity: usable data bytes. Must be a power of two. Pass 0 for default.
 * create:   1 = create (producer side), 0 = open existing (consumer side)
 *
 * Returns a heap-allocated RingBuffer* on success, NULL on failure.
 * Caller must call rb_destroy() when done.
 */
RingBuffer* rb_create(const char* name, uint32_t capacity, int create) {
    if (capacity == 0) capacity = DEFAULT_CAP;

    // Enforce power-of-two for the wrap mask trick
    if (capacity & (capacity - 1)) return nullptr;

    size_t total = HEADER_SIZE + capacity;

    int flags = create ? (O_CREAT | O_RDWR | O_TRUNC) : O_RDWR;
    int fd = shm_open(name, flags, 0600);
    if (fd < 0) return nullptr;

    if (create) {
        if (ftruncate(fd, (off_t)total) < 0) {
            close(fd);
            shm_unlink(name);
            return nullptr;
        }
    }

    void* mem = mmap(nullptr, total,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    if (mem == MAP_FAILED) {
        close(fd);
        if (create) shm_unlink(name);
        return nullptr;
    }

    auto* rb = new RingBuffer;
    rb->fd         = fd;
    rb->mem        = mem;
    rb->total_size = total;
    rb->hdr        = reinterpret_cast<RingBufferHeader*>(mem);
    rb->data       = reinterpret_cast<uint8_t*>(mem) + HEADER_SIZE;

    if (create) {
        memset(mem, 0, total);
        rb->hdr->magic    = MAGIC;
        rb->hdr->version  = VERSION;
        rb->hdr->capacity = capacity;
        rb->hdr->head.store(0, std::memory_order_release);
        rb->hdr->tail.store(0, std::memory_order_release);
        rb->hdr->dropped.store(0, std::memory_order_release);
    }

    return rb;
}

/*
 * rb_write — write a record to the ring buffer (producer side).
 *
 * Returns 1 on success, 0 if the buffer is full (record is dropped).
 */
int rb_write(RingBuffer* rb, const uint8_t* data, uint32_t len) {
    if (!rb || !data || len == 0) return 0;

    uint32_t record_size = sizeof(uint32_t) + len; // length prefix + payload
    if (rb_free(rb->hdr) < record_size) {
        rb->hdr->dropped.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }

    uint32_t cap  = rb->hdr->capacity;
    uint32_t tail = rb->hdr->tail.load(std::memory_order_relaxed);

    // Write length prefix (little-endian)
    uint8_t len_bytes[4] = {
        (uint8_t)(len & 0xFF),
        (uint8_t)((len >> 8) & 0xFF),
        (uint8_t)((len >> 16) & 0xFF),
        (uint8_t)((len >> 24) & 0xFF),
    };
    for (int i = 0; i < 4; ++i) {
        rb->data[tail & (cap - 1)] = len_bytes[i];
        tail++;
    }

    // Write payload, wrapping around as needed
    for (uint32_t i = 0; i < len; ++i) {
        rb->data[tail & (cap - 1)] = data[i];
        tail++;
    }

    rb->hdr->tail.store(tail, std::memory_order_release);
    return 1;
}

/*
 * rb_read — read one record from the ring buffer (consumer side).
 *
 * buf:      caller-supplied buffer for the payload
 * buf_len:  size of buf
 * out_len:  set to the actual payload length on success
 *
 * Returns 1 if a record was read, 0 if the buffer is empty or buf too small.
 */
int rb_read(RingBuffer* rb, uint8_t* buf, uint32_t buf_len, uint32_t* out_len) {
    if (!rb || !buf || !out_len) return 0;

    uint32_t used = rb_used(rb->hdr);
    if (used < sizeof(uint32_t)) return 0; // not even a length prefix yet

    uint32_t cap  = rb->hdr->capacity;
    uint32_t head = rb->hdr->head.load(std::memory_order_relaxed);

    // Peek at the length prefix
    uint32_t len = 0;
    for (int i = 0; i < 4; ++i)
        len |= ((uint32_t)rb->data[(head + i) & (cap - 1)]) << (8 * i);

    if (used < sizeof(uint32_t) + len) return 0; // payload not fully written yet
    if (len > buf_len) return 0;                  // caller's buffer too small

    head += sizeof(uint32_t);

    for (uint32_t i = 0; i < len; ++i) {
        buf[i] = rb->data[(head + i) & (cap - 1)];
    }
    head += len;

    rb->hdr->head.store(head, std::memory_order_release);
    *out_len = len;
    return 1;
}

/* rb_stats — fill in basic stats about the buffer state. */
void rb_stats(RingBuffer* rb, uint32_t* out_used, uint32_t* out_free,
              uint32_t* out_dropped) {
    if (!rb) return;
    if (out_used)    *out_used    = rb_used(rb->hdr);
    if (out_free)    *out_free    = rb_free(rb->hdr);
    if (out_dropped) *out_dropped = rb->hdr->dropped.load(std::memory_order_relaxed);
}

/* rb_destroy — unmap and close. Does not unlink the shm segment. */
void rb_destroy(RingBuffer* rb) {
    if (!rb) return;
    munmap(rb->mem, rb->total_size);
    close(rb->fd);
    delete rb;
}

/* rb_unlink — remove the named shared memory segment from the system. */
void rb_unlink(const char* name) {
    shm_unlink(name);
}

} // extern "C"