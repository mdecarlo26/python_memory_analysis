## Literature Review

This is the directory for all files and papers that are used for literview

### [Memray](https://github.com/bloomberg/memray)
This is written by fucking Bloomberg. [Docs](https://bloomberg.github.io/memray/tutorials/index.html)
This even goes down to the C level. It even track the python memory arena for python allocators. UGH
AND EVEN THREADS/SUBPROCESS. UGH
This looks like it does everything already. EMERGENCY MEETING

### [Scalene](https://github.com/plasma-umass/scalene)
This one is written by UMASS. THis even has GPU integration. looks pretty cool. It even does LLM based code optimizations.

### [Austin](https://github.com/P403n1x87/austin)
This samples the code in delta intervals and probes for memory differences between pings thus calculation approx memory usage.

### [YData](https://docs.profiling.ydata.ai/latest/)
For profiling large datasets sepcefically when using tools like Pandas. Seems to be more of an Exploratory Data Analysis package then memory profiling. Requires deeper look

## Overall View
No object based memory viewers/profilers. More along the line of a Heap analyzer not an allocation profiler. From random Hacker News user: "Allocation profilers will capture data about what is allocating memory over time. This can be captured in real time without interrupting the process and is usually relatively low-overhead.

Heap analyzers will generally take a heap dump, construct an object graph, do various analyses, and generate an interactive report. This generally requires that you pause a program long enough to create a heap dump, which is often multiple GB or more in size, write it to disk, then do the subsequent analysis and report generation."

## Possible Directions
Lifetime and Fragmentation profiling
GPU Profiling
