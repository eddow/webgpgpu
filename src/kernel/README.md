# kernel

For readability, the kernel code is divided in 3 files:
 - `io.ts` contain all the helpers functions to describe I/O generations
 - `scope.ts` creates the scope when `.kernel(...)` is called
 - `call.ts` the kernel function that is called to make the job