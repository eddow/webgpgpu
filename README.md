# WebGpGpu

This package provides WebGPU based GPU computing.

## Getting Started

### Installation

```bash
npm install webgpgpu
```

### Usage

```ts
import createWebGpGpu, { f32, threads } from 'webgpgpu'

async function main() {
	const webGpGpu = await createWebGpGpu()

	const kernel = webGpGpu
		.input({
			myUniform: f32,
			data: f32.array(threads.x)
		})
		.output({ produced: f32.array(threads.x) })
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
- TypeScript, and the whole is highly typed.
- Sizes assertion and even inference.
- Optimizations
  - buffer re-usage
  - workgroup-size calculation
  - `TypedArray` optimization js-side (only `set()` and `subarray()`)
  - etc.
- Compatibility: browser & node.js (*Many browsers still require some manipulation as WebGPU is not yet completely standardized)

## WebGPU code

Example kernel produced :
```wgsl
// #generated
@group(0) @binding(0) var<uniform> threads : vec3u;
@group(2) @binding(0) var<storage, read> a : array<f32>;
@group(3) @binding(0) var<storage, read_write> outputBuffer : array<f32>;
// #user-defined

fn myFunc(a: f32, b: f32) -> f32 { return a + b; }

// #generated
@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) thread : vec3u) {
	if(all(thread < threads)) {
// #user-defined

		outputBuffer[thread.x] = myFunc(a[thread.x], 3.);

// #generated
	}
}
```

The 2 reserved variables are `thread` (the `xyz` of the current thread) and `threads` (the size of all the threads). There is no workgroup interaction for now.

Pre-function code chunks can be added freely (the library never parses the wgsl code) and the content of the (guarded) main function as well

## WebGpGpu class

The `WebGpGpu` class exposes a `root` that promises an instance and allows to create sub-instances by specification (the values are never modified as such), so each specification code indeed creates a new instance who is "more specific" than the parent.

### kernel

This is the only non-chainable function : creates a kernel (in javascript, a function) that can be applied on the inputs. It takes the main code (the one of the main function) as argument.

```ts
const kernel = webGpGpu
	...
	.kernel(/*wgsl*/`
outputBuffer[thread.x] = a[thread.x] * b;
	`)
```

> Note: The kernel function retrieves the *whole* generated code on `toString()`

### defined

Adds a chunk of code to be inserted before the main function. Plays the role of `#define` and `#include`.

```ts
webGpGpu.defined(/*wgsl*/ `
fn myFunc(a: f32, b: f32) -> f32 { return a + b; }
`)
```

### input

Declares inputs for the kernel. Takes an object `{name: type}`.

```ts
webGpGpu.input({
	myUniform: f32,
	data: f32.array(threads.x),
	randoms: i32.array(133)
})
```

### output

Declares outputs for the kernel. Takes an object `{name: type}`.

```ts
webGpGpu.output({
	produced: f32.array(threads.x)
})
```

### common

Defines a common input value to all calls (and makes a unique transfer to the GPU)

```ts
const kernel = webGpGpu
	.input({ b: f32.array(threads.x) })
	.common({ a: f32.array(threads.x).value([1, 2, 3]) })
	.output({ output: f32.array(threads.x) })
	.kernel('outputBuffer[thread.x] = a[thread.x] + b[thread.x];')
const { output } = await kernel({b: [4, 5, 6]})	// output ~= [5, 7, 9]
```


### workGroup

If you know what a workgroup is and really want to specify its size, do it here.

```ts
webGpGpu.workGroup(8, 8)
```

## Types

The main types from wgsl are available with their wgsl name (`f32`, `vec2f`, etc.). Note: These are *values* who specify a wgsl *type* - it is not a typescript type. These  types (like `Input1D<number, Float32Array>`) are produced and used automatically.

Types can be array-ed. Ex:
```ts
f32.array(3)
f32.array(3).array(4)
//or
f32.array(3, 4)
```

Multi-dimensional arrays are represented in the shader as `texture_2d` or `texture_3d`. They support only elements of size 1, 2 or 4.
Ex.: `f32`, `vec2u`, `mat2x2f`

Arguments (simple, arrays of any dimension) can always be passed as corresponding `TypedArray`. So, `mat3x2f.array(5).value(new Float32Array(3*2*5))` is doing the job! (even if array sizes are still validated)

Types also specify how to read/write elements from/to a `TypedArray` (`Float32Buffer`, ...).
 - most (all `vec...` and `mat...`) *do not* copy for read, `subarray` is used, so each element is an `ArrayLike` (who access directly the middle of the `Xxx##Array`)
 - singletons just set/get from the TypedArray without transformation

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

Hence, in order to know if it's supported, `webGpGpu.f16` tells if it exists and all the f16 types (`f16`, `vec2h`, `mat2x2h`, etc.) will be set to their `f32` equivalent until when the first `WebGpGpu` is ready and confirms their availability.

The system has not yet been completely tested and remains the question of writing f16 immediate values &c.

### "Type" objects

These types object offer (if needed) these functions. The functions changing the definition are chainable and have no side effect, they create a new type object from the original one and the given specifications.

#### array

Declares an `array of` something. Ex: `f32.array(3, 4)`

#### transform

Declares a transformation to read/write `TypedArray`.
- `elementConvert` gives an array-like (perhaps directly a `TypedArray`) from a value
- `elementRecover` creates a value from an array-like (a `TypedArray.subarray`)

Note that arrays can be converted! The next code is doing the expected job : `MyClass` is transferable to an array of float (hence the transform function have a 1-D size parameter) and an array

```ts
const myValues: MyClass[] = [...]
const myType = f32.array(threads.x).transform<MyClass>(
	(m: MyClass, [size]: [number])=> [...],
	(o: ArrayLike<number>, [size]: [number])=> new MyClass(o)
)
myType.array(threads.y).value(myValues)
```

#### value

Just creates a "typed value" (ex: `f32.value(1)`) that can be used as argument of many WebGpGpu functions.

The given value can always be a `FlatSomethingArray` (`TypedArray`) or some combination (array of D-1 inputs).

> Not chainable! A "typed value" is not a type. It wraps it as buffable in `{ buffable, value }`

#### toTypedArray

Creates a `TypedArray` from the given value. It also manages the size inference.

```ts
toTypedArray(
	workSizeInfer: WorkSizeInfer,
	data: InputXD<OriginElement, InputSpec, Buffer>,
	required?: string
)
```

## Size inference

`threads.x`, `threads.y` and `threads.z` are internal values (symbols) that can be used as "jokers" - or not. Each `WebGpGpu` has a state of inference (it knows some sizes and others not) and will infer them as soon as possible.

1. The first inference will be when specifying commons (define-time)
2. The second inference will be when specifying inputs (run-time)
3. The remaining can be given as supplementary arguments of `kernel` (define-time, but these arguments will be used *after* input inference)
4. Lastly, supplementary arguments can be given *to* the produced kernel
5. Finally, it is just supposed `1`

In 3. & 4., the last arguments if specified are:
- `{x?: number, y?: number, z?: number}` the values to default to/require
- A string combination of 'x', 'y' and 'z' ('z', 'xy') specifying which ones of them are required (otherwise they are ignored if not needed). Defaults to `''`

Generate the 10 first squares:
```ts
const kernel = webGpGpu
	.output({output: f32.array(threads.x)})
	.kernel('outputBuffer[thread.x] = thread.x*thread.x;', { x: 10 })
const { output } = await kernel({})
```

Generate the N first squares:
```ts
const kernel = webGpGpu
	.output({output: f32.array(threads.x)})
	.kernel('outputBuffer[thread.x] = thread.x*thread.x;')
const { output } = await kernel({}, { x: 10 })
```

## Structures

TODO

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
import createWebGpGpu from 'webgpgpu'

async function main() {
	const webGpGpu = createWebGpGpu({}, {}, 'enable-dawn-features=allow_unsafe_apis,dump_shaders,disable_symbol_renaming', ...)
	...
}
```

#### Hand-made

If you manage to have your own, want to share a device, ...
`WebGpGpu` exposes :
```ts
class WebGpGpu {
	static createRoot(root: GPUDevice, options?: { dispose?: () => void }): WebGpGpu
	static createRoot(
		root: GPUAdapter,
		options?: { dispose?: () => void; deviceDescriptor?: GPUDeviceDescriptor }
	): Promise<WebGpGpu>
	static createRoot(
		root: GPU,
		options?: {
			dispose?: () => void
			deviceDescriptor?: GPUDeviceDescriptor
			adapterOptions?: GPURequestAdapterOptions
		}
	): Promise<WebGpGpu>

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

`warn` and `error` can be set separately as the whole library uses it. (mainly for compilation messages) or extreme cases as "uploaded size(0) array", ...
Note that a `log.error` will always have its associated exception throw.

## Exceptions

- `CompilationError` Has the exact messages in the `cause` (they are also logged)
- `ArraySizeValidationError` Occurs when arguments size are not fitting
- `ParameterError` Mainly for parameter names conflicts &c.

## Ecosystem

- [VS Code plugin for inline-wgsl coloring](https://marketplace.visualstudio.com/items?itemName=ggsimm.wgsl-literal)
- Linux: Do *not* use chromium, it will not support WebGPU - install chrome/firefox(untested)/...

## TODOs

- Texture management (for now, only 0/1-D)
- structures
- Perhaps some canvas output ?
- Chaining (output become input of next kernel without transfer in CPU memory)
  - Even more complex pipeline management?