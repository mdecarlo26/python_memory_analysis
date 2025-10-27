# Project ideas based off new info

## P1 - Unified GPU and CPU Memory  
* Can use Pytorch pre-built `Memory Snapshot` code stack.
* Use possible solution from `P2`
* Need someway to link the two.

```{python}
arr1

arr1.to(device) # snoop on bus send
```

## P2 - Object Level Memory Utilization (CPU only)
* Build an Object Graph and attribute memory size to each object
* Visuals
  - Sort by most memory objects

## P3 - All of the above
* No idea how much work each takes

## P4 - How do we look at object memory!
* Has this been done? and how