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

### [PyTorch GPU Monitoring](https://pytorch.org/blog/understanding-gpu-memory-1/)
Pytorch exposes `Memory Snapshot` tool for monitoring GPU callstack and memory profile specifically for Out Of Memory (OOM) errors. This seems to be total board memory usage but also follows the call stack.
[REPO](https://github.com/pytorch/pytorch.github.io/tree/site/assets/images/understanding-gpu-memory-1)


### [VScode Extension API](https://code.visualstudio.com/api/get-started/your-first-extension)
Getting started page for making VScode extensions. Basic set-up for creating extensions. Uses a package called [Yeoman](https://yeoman.io/) to scaffold projects or useful parts. Then can use typescript or whatevr you prefer to build your application.
  
#### Three Main Parts to VScode Extensions:
* Activation Events: ebvents upon which your extension becomes active
* Contribution Points: static declarations that you make in the package.json Extension Manifest to extend VS Code.
* VS Code API: a set of JavaScript APIs that you can invoke in your extension code.

#### [Web Extensions](https://code.visualstudio.com/api/extension-guides/web-extensions)
Here is a guide for creating a web based VScode Extensions. Limited by a browser sandbox compared to normal extensions. It does have full access to the VS Code API.
 
## Overall View
No object based memory viewers/profilers. More along the line of a Heap analyzer not an allocation profiler. From random Hacker News user: "Allocation profilers will capture data about what is allocating memory over time. This can be captured in real time without interrupting the process and is usually relatively low-overhead.

Heap analyzers will generally take a heap dump, construct an object graph, do various analyses, and generate an interactive report. This generally requires that you pause a program long enough to create a heap dump, which is often multiple GB or more in size, write it to disk, then do the subsequent analysis and report generation."

## Possible Directions
Lifetime and Fragmentation profiling
GPU Profiling
