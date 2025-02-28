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

Inferences are `Record<string, number|undefined>` where a number is a specification and `undefined` a simple declaration.

To add bindings, a class should extend `Bindings<NeededInferences>`. The `NeededInferences` inference type should be provided (or `{}`)
