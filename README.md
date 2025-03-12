![npm](https://img.shields.io/npm/v/webgpgpu.ts)

# WebGpGpu.ts

This package provides WebGPU based GPU computing.

versions:
 - 0.1.x: beta

## Getting Started

The complete documentation is available on [github pages](https://eddow.github.io/webgpgpu/).

### Installation

```bash
npm install --save webgpgpu.ts
```

### Usage

```ts
import createWebGpGpu, { f32 } from 'webgpgpu.ts'

async function main() {
	const webGpGpu = await createWebGpGpu()

	const kernel = webGpGpu
		.input({
			myUniform: f32,
			data: f32.array('threads.x')
		})
		.output({ produced: f32.array('threads.x') })
		.kernel(/*wgsl*/`
	produced[thread.x] = myUniform * data[thread.x];
		`)

	const { produced } = await kernel({
		myUniform: 2,
		data: [1, 2, 3, 4, 5]
	})
	// produced -> [2, 4, 6, 8, 10]
}
```

### Presentation

Basically, WebGpGpu manages purely `compute` shaders in order to make in-memory GPU computing possible.

The GPU parallelize loops that would be here standardized like
```js
for (thread.x = 0; thread.x < threads.x; thread.x++) {
	for (thread.y = 0; thread.y < threads.y; thread.y++) {
		for (thread.z = 0; thread.z < threads.z; thread.z++) {
			/* here */
		}
	}
}
```

The point of the library is to automatize the parallelization and all the configurations and concepts and learning curve that usually come with it.
For those who tried a bit, all the bindings, buffer writing/reading, and other things that are necessary to write a GPU program, are hidden from the user.

With real pieces of :
- TypeScript, as the whole is highly typed.
- Sizes assertion and even inference.
- Optimizations
  - buffer re-usage
  - workgroup-size calculation
  - `ArrayBuffer` optimization js-side (no superfluous read/writes, ...)
  - etc.
- Compatibility:
  - browser: Many browsers still require some manipulation as WebGPU is not yet completely standardized
  - node.js through the library [node-webgpu](https://github.com/dawn-gpu/node-webgpu)

## WebGPU code

Example kernel produced :
```rust
// #generated
@group(0) @binding(0) var<storage, read> a : array<mat2x2f>;
@group(0) @binding(1) var<storage, read> b : array<mat2x2f>;
@group(0) @binding(2) var<uniform> threads : vec3u;
@group(0) @binding(3) var<storage, read_write> output : array<mat2x2f>;

// #user-defined

fn myFunc(a: mat2x2f, b: mat2x2f) -> mat2x2f {
	return a + b;
}

// #generated
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
// #user-defined

		output[thread.x] = myFunc(a[thread.x], b[thread.x]);

// #generated
	}
}
```

The 2 reserved variables are `thread` (the `xyz` of the current thread) and `threads` (the size of all the threads). There is no workgroup interaction for now.

Pre-function code chunks can be added freely (the library never parses the wgsl code) and the content of the (guarded) main function as well