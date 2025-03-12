# Bindings

Bindings allow to specify how the data is passed between host and device.

A `Bindings` class defines a a set of bindings, how they are provided to the device and how they are recovered, along with inferences interactions.

## Provided bindings

### CommonBindings

`CommonBindings(Record<string, ValuedBuffable<Inferences>)`: Defines common bindings for all kernels. The keys match names of the variables defined by the shader program.

```ts
const common = new CommonBindings({
	a: f32.array('threads.x').value([1, 2, 3])
});
```

### InputBindings

`InputBinding(Record<string, Buffable>)`: Defines an input binding. The keys must match the names of the inputs defined by the shader program.

Defines the inputs the kernel can take.

```ts
const input = new InputBinding({ a: f32.array('threads.x') });
```

### OutputBindings

`OutputBinding(Record<string, Buffable>)`: Defines an output binding. The keys must match the names of the outputs defined by the shader program.

Defines outputs the kernel can produce.

```ts
const output = new OutputBinding({ b: f32.array('threads.x') });
```

### InferenceBindings

```ts
InferenceBindings<Record<
	string,
	| Inferred
	| readonly [Inferred, Inferred]
	| readonly [Inferred, Inferred, Inferred]
	| readonly [Inferred, Inferred, Inferred, Inferred]
>>
```

Define inferences that can therefore be used both in the wgsl code and as array sizes. 

## Custom bindings

Creating a specific Bindings class allows to define custom bindings. It can define new inferences, inputs and outputs

## Functionment

### Construction/attachment

Even if a binding is made to be attached to one WebGpGpu instance, it is constructed independently *then* attached to it.

The object then has its method `init` when attached once it is provided with a `device`. If overridden, this function should return an array of:
```ts
interface BindingEntryDescription {
	declaration: string
	layoutEntry: GPUUnboundGroupLayoutEntry
}

/*where*/ type GPUUnboundGroupLayoutEntry = Omit<GPUBindGroupLayoutEntry, 'binding'>
```
where `declaration` is the WGSL declaration of a variable and `layoutEntry` is the layout entry corresponding to it.

#### Produced inferences

On construction, inferences can be generated as `Record<string, number|undefined>` where a number is a specification and `undefined` a simple declaration.

To add bindings, a class should extend `Bindings<NeededInferences>`. The `AddedInferences` inference type should be provided (or `{}`) by the member `addedInferences` (who should then be typed appropriately - `{}` if no added inferences).

### entries

The entries are generated at each call, when inputs are provided. These inputs are provided to the `entries` method which returns an array of:
```ts
type GPUUnboundGroupEntry = Omit<GPUBindGroupEntry, 'binding'>
```
which will be bound to the appropriate location on the device.

```ts
entries(
		inputs: {},
		inferences: AnyInference,
		reasons: Record<string, string>
	): GPUUnboundGroupEntry[]
```
Takes the inputs *whose types should be specified here*, as the type of the first argument will be used to infer the inputs accepted by the kernel.

Note: every binding receive every inputs of the kernel' call.

### encoder/read

`encoder` is called given an input and an encoder. This encoder is to be used for read operations from the GPU.
```ts
encoder(inputs: {}, commandEncoder: GPUCommandEncoder): void
```

`read` is called at the end of the execution of the kernel. It takes the input object and returns a promise of outputs. This function should have a good return value type as this one will be used to infer the outputs of the kernel.
```ts
read(inputs: {}): Promise<MyOwnOutputs>
```

#### Inputs and WeakMaps

The best way for concurrence is to have call-dependant values stored in a `WeakMap` whose key is the input object - as this is the constant `entries`, `encoder` and `read` get on a same kernel call. `OutputBindings` shows an example of such implementation.