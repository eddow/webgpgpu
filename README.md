TODO:
- no more defaults
- batch

![npm](https://img.shields.io/npm/v/webgpgpu.ts)

# WebGpGpu

This package provides WebGPU based GPU computing.

versions:
 - 0.0.x: alpha

## Getting Started

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

## WebGpGpu class

A main function allows to create a root `WebGpGpu`: `const webGpGpu = await createWebGpGpu()` that allows to create sub-instances by specification (the values are never modified as such), so each specification code indeed creates a new instance who is "more specific" than the parent.

### kernel

This is the only non-chainable function : creates a kernel (in javascript, a function) that can be applied on the inputs. It takes the main code (the one of the main function) as argument.

```ts
const kernel = webGpGpu
	...
	.kernel(/*wgsl*/`
output[thread.x] = a[thread.x] * b;
	`)
```

> Note: The kernel function retrieves the *whole* generated code on `toString()`

### define & import

Adds a chunk of code to be inserted before the main function. Plays the role of `#define` and `#include`. They use a structure with optionals `declaration` and `computation`. The former is added outside the function, the latter inside the main function, before the main code

- direct definition
```ts
webGpGpu.define({
	declaration: /*wgsl*/`
fn myFunc(a: f32, b: f32) -> f32 { return a + b; }
`
	})
```

- non-repeating usage
`WebGpGpu` has a static property `imports` that is editable at will and just contain a named collection of code chunks. The function `webGpGpu.import(...)` can be used with the key of such import making sure the import will be included once.

### workGroup

If you know what a workgroup is and really want to specify its size, do it here.

```ts
webGpGpu.workGroup(8, 8)
```

## Bindings

These functions are shortcuts to [`Bindings`](./src/binding/README.md) creation and are chainable.

Example of equivalence:
```ts
webGpGpu.input({a: f32})

webGpGpu.bind(inputs/*->InputBindings*/({a: f32}))
```

### input

Declares inputs for the kernel. Takes an object `{name: type}`.

```ts
webGpGpu.input({
	myUniform: f32,
	data: f32.array('threads.x'),
	randoms: i32.array(133)
})
```

### output

Declares outputs for the kernel. Takes an object `{name: type}`.

```ts
webGpGpu.output({
	produced: f32.array('threads.x')
})
```

### common

Defines a common input value to all calls (and makes a unique transfer to the GPU)

```ts
const kernel = webGpGpu
	.input({ b: f32.array('threads.x') })
	.common({ a: f32.array('threads.x').value([1, 2, 3]) })
	.output({ output: f32.array('threads.x') })
	.kernel('output[thread.x] = a[thread.x] + b[thread.x];')
const { output } = await kernel({b: [4, 5, 6]})	// output ~= [5, 7, 9]
```

### infer & specifyInference

`infer` allows to create an inference (cf. [Size inference](#size-inference) section).

```ts
webGpGpu
	.infer({ myTableSize: [undefined, undefined] })
	.input({ myTable: f32.array('myTableSize.x', 'myTableSize.y') })
```

With this code, the variable `myTableSize` will be a `vec2u` available in the wgsl code that will be fixed (here, when a `myTable` of a certain size will be given as argument)

To fix (assert) an existing inference, `specifyInference` can be used.
```ts
webGpGpu.specifyInference({ 'myTableSize.x': 10 })
```

## Kernel

The kernel is the function that takes the input and returns (a `Promise` of) the output(s).

```ts
const kernel = webGpGpu
	.input({ a: f32.array('threads.x'), b: f32.array('threads.x') })
	.output({ output: f32.array('threads.x') })
	.kernel('output[thread.x] = a[thread.x] + b[thread.x];')
const { output } = await kernel({ a: [1, 2, 3], b: [4, 5, 6] }) // output ~= [5, 7, 9]
```

### Calling

The kernel can take as a second argument an object containing defaults for the inferences. These values *will not* be forced/asserted and might not be used. See [Size inference](#size-inference) section.

### Inputs

Inputs are given as an object `{name: value}`. Values can be either an `ArrayBufferLike` or
- Their element if not an array (`D = 0`), like a number, a triplet of vector (depending on the type used)
- An array of dimension `D - 1` inputs when it is an array of some dimension (`D > 0`).

### Outputs

The given values is a dynamic `ArrayBuffer`-reader that act as JS arrays. The `operator[](index: number)` is hacked in and the array interface will be forwarded.

> Note: There is no array creation so to speak while not specifically asked for, it all end up being an access to the underlying `ArrayBuffer`.

## Types

The main types from wgsl are available with their wgsl name (`f32`, `vec2f`, etc.). Note: These are *values* who specify a wgsl *type* - it is not a typescript type. These  types (like `Input1D<[number, number]>`) are produced and used automatically (here, from a `vec2f.array(x)`).

Arguments (simple, arrays of any dimension) can always be passed as corresponding `ArrayBuffer`. So, `mat3x2f.array(5).value(Float32Array.from([...]))` is doing the job! (even if array sizes are still validated)

Types also specify how to read/write elements from/to an `ArrayBuffer`.
 
For convenience, these types have been added:
- `Vector2`
- `Vector3`
- `Vector4`
- `RGB`
- `RGBA`

These actually encode/decode in order to use their respective interface, ex. `{x: number, y: number}` for `Vector2`.
These "shaped" types use `f16` for the precision

### f16

16-bit float is a thing in gpus and should be taken into account as it's a bit the "native" or "optimized" work size (important when working with mobile devices for ex). The big draw back is that *all devices don't support it*.

Hence, in order to know if it's supported, `webGpGpu.f16` tells if it exists and all the f16 types (`vec2h`, `vec3h` and `vec4h`) will be set to their `f32` equivalent until when the first `WebGpGpu` is ready and confirms their availability.

The system has not yet been completely tested and remains the question of writing f16 immediate values &c.

### "Type" objects

These types object offer (if needed) these functions. The functions changing the definition are chainable and have no side effect, they create a new type object from the original one and the given specifications.

#### array

Declares an `array of` something. Ex:
```ts
f32.array(3)
f32.array(3).array(4)
//or
f32.array(4, 3)	// take care .array(X).array(Y) -> .array(Y, X)
```

In all array accesses in TS, the multi-dimensional indexes are given *most-important first*.
- `f32.array(3).value([1, 2, 3])`
- `f32.array(2, 3) -> f32.array(3).array(2)` 
  - `value([[1, 2, 3], [4, 5, 6]]).at(1, 2) === 6`
  - `value([[1, 2, 3], [4, 5, 6]]).slice(0) ~ [1, 2, 3]`
- `f32.array(3, 2) -> f32.array(2).array(3)`
  - `value([[1, 2], [3, 4], [5, 6]]).at(2, 1) === 6`
  - `value([[1, 2], [3, 4], [5, 6]]).slice(0) ~ [1, 2]`

In WGSL, a "stride" is computed and accessible in the whole code (as `var<private>` for now - 0.0.7) named after the wgsl name of the value (input/output/...) post-fixed with `Stride`

eg: 
```ts
input({ myTable: f32.array('threads.y', 'threads.x') }).kernel(...)
```
can be indexed in the wgsl code with:
```rust
let entry = myTable[dot(thread.yx, myTableStride)];
```

> Note: It is advised to keep threads.x as the last (right-most, least-significant) index of the array.

#### value

Just creates a "typed value" (ex: `f32.value(1)`) that can be used as argument of many WebGpGpu functions.

Ways to give the value happen the same as for the [inputs](#inputs).

> Not chainable! A "typed value" is not a type. It wraps it as buffable in `{ buffable, value }`

## Size inference

One inference exists in all computation: `threads: vec3u`, but others can be declared and used.

When sizes are specified - bound as commons or given as inputs, an inference can be used - the WebGpGpu engine remembers an inferring status (what is known what is not), deduce from given arrays and assert sizes.

In the shader code, inferences can be used directly (they are declared in their `u32` shade) and the values will be provided as uniforms.

Inferences are meant to replace `arrayLength` and other mechanism. If really a random-size table has to be given and its size retrieved, this can be used:
```ts
webGpGpu
	.infer({ myTableSize: [undefined, undefined] })
	.input({ myTable: f32.array('myTableSize.x', 'myTableSize.y') })
```
and `myTableSize` will be a provided `vec2u`.

### Inference declarations and value forcing

cf. [infer & specifyInference](#infer--specifyinference)

### Inference defaulting

If inferences cannot be retrieved from an array size, they can be defaulted to a number (or will default to `1`) when defining or calling the kernel.

Note: This defaulting system doesn't assert anything and will perhaps not even be taken into account if the value was already inferred. To force a value, use `.infer`.

Generate the N first squares:
```ts
const kernel = webGpGpu
	.output({output: f32.array('threads.x')})
	.kernel('output[thread.x] = thread.x*thread.x;')
const { output } = await kernel({}, { 'threads.x': 10 })
```

Generate the N(default 10) first squares:
```ts
const kernel = webGpGpu
	.output({output: f32.array('threads.x')})
	.kernel('output[thread.x] = thread.x*thread.x;', { 'threads.x': 10 })
const { output } = await kernel({})
```

## System calls

### Creation

The library exposes a function `createWebGpGpu` that creates a root `WebGpGpu` object.
```ts
function createWebGpGpu(
	adapterOptions?: GPURequestAdapterOptions,
	deviceDescriptor?: GPUDeviceDescriptor,
	[...WebGPUOptions: string[]]
)
```

#### Node.js only

The `WebGPUOptions` are only available to the node.js clients.
The library uses [node-webgpu](https://github.com/dawn-gpu/node-webgpu) who allows giving parameters when creating the GPU object. These parameters can be given to the default creation export.

```ts
import createWebGpGpu from 'webgpgpu.ts'

async function main() {
	const webGpGpu = createWebGpGpu({}, {}, 'enable-dawn-features=allow_unsafe_apis,dump_shaders,disable_symbol_renaming', ...)
	...
}
```

#### Hand-made

If you manage to have your own adapter/device, want to share a device, ...
`WebGpGpu` exposes :
```ts
class WebGpGpu {
	static createRoot(device: GPUDevice, options?: { dispose?: () => void }): RootWebGpGpu
	static createRoot(
		adapter: GPUAdapter,
		options?: { dispose?: (device: GPUDevice) => void; deviceDescriptor?: GPUDeviceDescriptor }
	): Promise<RootWebGpGpu>
	static createRoot(
		gpu: GPU,
		options?: {
			dispose?: (device: GPUDevice) => void
			deviceDescriptor?: GPUDeviceDescriptor
			adapterOptions?: GPURequestAdapterOptions
		}
	): Promise<RootWebGpGpu>

	get device(): GPUDevice
	dispose(): void
}
```

Note: the `dispose` function disposes the all the WebGpGpu objects from the root (created by `createWebGpGpu` or `WebGpGpu.createRoot`)

### Logging

`WebGpGpu` exposes:
```ts
WebGpGpu.log: {
	warn(message: string): void,
	error(message: string): void,
}
```

`warn` and `error` can be set separately to redirect the whole library logs. (mainly for compilation messages) or extreme cases as "uploaded size(0) array", ...
Note that a `log.error` will always have its associated exception throw.

## Exceptions

- `CompilationError` Has the exact messages in the `cause` (they are also logged)
- `ArraySizeValidationError` Occurs when arguments size are not fitting
- `ParameterError` Mainly for parameter names conflicts &c.

## Ecosystem

- Configured VSCode plugins:
  - [karma test explorer](https://marketplace.visualstudio.com/items?itemName=lucono.karma-test-explorer)
  - [mocha test explorer](https://marketplace.visualstudio.com/items?itemName=hbenl.vscode-mocha-test-adapter)
- Other useful VSCode extensions:
  - [inline-wgsl syntax highlighting](https://marketplace.visualstudio.com/items?itemName=ggsimm.wgsl-literal)

### Ubuntu

Do *not* use chromium, it will not support WebGPU - install chrome/firefox(untested)/...

```bash
# 1. Add Google Repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | \
sudo tee /etc/apt/sources.list.d/google-chrome.list

# 2. Add Google Signing Key
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg

# 3. Update and Install
sudo apt update
sudo apt install google-chrome-stable

google-chrome
```

## TODOs & limitations

### Limitations

The main limitation is WebGPU support.

It is supported in some browsers but poorly support automated testing.

For node, this library uses [node-webgpu](https://github.com/dawn-gpu/node-webgpu) who is really fresh and does not yet allow a smooth ride for all cases (automated testing is possible in some specific circumstances)
For instance, for now, a complete mocha testing run is impossible: some async fixture system:error or something else breaks - but tests are usable few by few

### Roadmap

- Structures and automatic organization for size optimization
- UBO creation: for now, a single `f32` as input *is* an UBO. We need UBO (and their types) built automatically 
  - CODE PARSING! replace `myUniform` by `UBO0.myUniform`
- Automatic array strides computations : now var<private> -> uniform ?
  - Code parsing: allow some operators for like `myArray[<myVec2Index>]` -> `myArray[dot(myVec2Index, myArrayStride)]`
- Arrays position optimization:
  - When possible, use fixed-size arrays (if size is completely inferred at layout time)
  - If such happen, have stride object given as const, not uniform
  - Check GPU limitations to have input arrays with fixed-size small enough given directly in the UBOs

#### Parallel

- Size assertion/inference when ArrayBuffers are provided directly as X-D inputs (X > 1) 
- Make BufferReader more ArrayLike (iterator, array prototype forward, ...)
- Code parsing: `f16` replacement: for immediate values:
  - `###h` -> `###f` (with `###` being a valid number)
  - `vec2h`, `vec3h`, `vec4h` -> `vec2f`, `vec3f`, `vec4f`: for now it happens in the JS declarations, so the generated code - but not if used directly in the code
- Some wizardry à la gpu.js (js-> wgsl) ?