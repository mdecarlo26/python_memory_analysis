# Information about Python Garbage Collector
The Garbage Collector will be known as GC for the rest of the doc. The GC is exposed in Python in the `gc` module.

## Reference Counting [Official Docs](https://docs.python.org/3/extending/extending.html#reference-counts)
Each object has a counter which is incremented when the object is referenced somewhere (stored in a variable, in a list, etc) and dereferenced when the ref is deleted (pop from list, function call return, etc). Zero reference count leads to object deletion and memory free.
### Cycle Detector
The GC also has a detector to prevent circular references from spawning. Can be direct or indirect references. Cycle detector is exposed in the `gc` module through `collect()`.


