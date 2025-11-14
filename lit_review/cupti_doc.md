# Cupti Documentation
---

**All Information Sourced from: [NVIDIA](https://docs.nvidia.com/cupti/index.html)**

## Useful Terms
* Tracing - refers to the collection of timestamps and additional information for CUDA activities such as CUDA APIs, kernel launches and memory copies during the execution of a CUDA application. Tracing helps in identifying performance issues for the CUDA code by telling you which parts of a program require the most time. Used with the Activty & Callback API

* Profiling - the collection of GPU performance metrics for individual kernels, or a set of kernals in isolation. Metrics can be collected with the **CUPTI Range** & **Host Profiling** APIs

---

## Activity API
==Main Purpose: record and deliver detailed performance and memory activity data from CUDA programs, through the use of timestamped records of runtime activities.==

* Use cases:
  * Memory information like, `cudaMalloc` and `cudaFree`
  * tracking memory allocations, free's, and copies
  * Correlating CPU/GPU events
  * Analyzing Kernal Execution behavior
    * Kernal start and end times
    * execution duration
    * Device IDs
    * Streams and Concurrency

* Terminology:
  * Activity Record
    * CPU and GPU activity is reported in C data structures called **Activity Records**
    * there are different kinds of C Structs for each activity kind
    * records are referred to using `CUpti_Activity` type, which only has one field that is the kind of activity record. You can use this to cast to the specific type that represents the desired activity

  * Activity Buffer:
    * Used to transfer one or more acitvity records from CUPTI to the client.
    * Cupti fills activity buffers with activity records as the specified activity occurs or CPU and GPU
    * Doesn't guarentee any ordering
    * Client is responsible for providing empty activity buffers
    * Uses callbacks to request and return buffers of activity records
    * There is also an asyncronous buffering API, which you must register 2 callbacks using the function `cuptiActivityRegisterCallbacks`. One of the callbacks will be used when CUPTI needs an empty buffer, and the other delivers a buffer containing one or more activity records to the client.
    * Recommended that buffer size is between 1-10MB

    * Client can make a request to deliver buffers at anytime using `cuptiActivityFlushAll` and `cuptiActivityFlushPeriod`
      * the `FlushAll` version (with flag set to `0`) returns all activity buffers that have all the activity records **completed**; however, they **don't** need to be full. Won't return buffers that have **any** incomplete records.
        * there is another flag, `CUPTI_ACTIVITY_FLAG_FLUSH_FORCED`, which included the incomplete records.
      * the `FlushPeriod` version, CUPTI only returns buffers that are full and fully complete.
---

## Activity Kinds
Each activity kind corresponds to a specific category of CUDA events.

### Common Types
- **`CUPTI_ACTIVITY_KIND_MEMORY`** – GPU memory allocations/frees
- **`CUPTI_ACTIVITY_KIND_MEMCPY`** – Memory copies (H↔D, D↔D)
- **`CUPTI_ACTIVITY_KIND_KERNEL`** / **`CONCURRENT_KERNEL`** – Kernel metadata
- **`CUPTI_ACTIVITY_KIND_RUNTIME`** / **`DRIVER`** – Runtime & driver API calls
- **`CUPTI_ACTIVITY_KIND_DEVICE`** / **`CONTEXT`** – Device/context metadata

---

## How to Use the Activity API

### 1. Register Callbacks
```c
cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
```

### 2. Enable Activity Kinds
```c
cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY);
cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
```

### 3. Run CUDA Workload
CUPTI logs activities asynchronously during execution.

### 4. Process Activity Records
Inside `bufferCompleted`:
```c
cuptiActivityGetNextRecord(buffer, validSize, &record);
```
Cast based on `record->kind`.

### 5. Flush Remaining Buffers
```c
cuptiActivityFlushAll(0);
```

### 6. Disable Activity Kinds (Optional)
```c
cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY);
```

---

## What Information You Can Extract

### Memory Behavior
- Allocation sizes & addresses
- Frees / leak detection
- Unified memory events
- Copy sizes and directions
- GPU memory usage timeline

### Kernel Execution
- Start & end timestamps
- Duration
- Grid/block dimensions
- Registers & shared memory
- Stream ID / concurrency

### CPU/GPU Correlation
- API-call latency
- CPU thread ↔ GPU kernel relationships
- Blocking vs asynchronous calls

---

## How we can use it
- Enables **real-time GPU memory visualization**
- Allows tracking of device memory usage for Python workloads
- Helps diagnose **GPU memory leaks**
- Allows correlation between:
  - **Python object graphs** (via `objgraph`)
  - **GPU memory activity** (via CUPTI)
- Forms the foundation for a unified **CPU + GPU memory analysis tool**


---

## How to use Callback API
Allows you to register a callback into your own code.

* Terminology
  * Callback Domain
  * Callback ID
  * Callback Function
  * Subscriber