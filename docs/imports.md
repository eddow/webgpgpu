# Imports

Named imports can be done with `#imports` or simply `webGpGpu.import('import1', 'import2', ...)`.

They are meant to be imported *once* at the end of the day no matter how many time they have been imported and to be imported "in order"

## Source of imports

The import source is really up to the developer. A static function `WebGpGpu.defineImports({ import1: code, ... })` allows to dynamically define them.

A good way to do it when using vite is:
```ts
const shaderImports = import.meta.glob(`${wgslFolder}/*.wgsl`, { query: '?raw', eager: true })
for (const path in shaderImports)
	shaders[/\/([^\/]*)\.wgsl/.exec(path)![1]] = (shaderImports[path] as any).default as string
WebGpGpu.defineImports(shaders)
```

## CodeParts

```ts
interface CodeParts {
	imports?: Iterable<PropertyKey>
	declaration?: string
	initialization?: string
	computation?: string
	finalization?: string
}
```

Imports are defined with code parts.

- `imports` They can simply other imports - circular references will be reported by a `CircularImportError` when the imports are resolved (on code generation).

As these imports are used in a shader (declaring a function), in the case `childImport` imports `parentImport` and is imported in a code generation, then here is how the generated code will look like:

```
...
parentImport.declaration
childImport.declaration
...
fn main(...){
	...
	parentImport.initialization
	childImport.initialization
	...
	parentImport.computation
	childImport.computation

	... // Main code given to compute, the argument of `kernel` for example.
	
	childImport.finalization
	parentImport.finalization
	...
}
```

## Pre-processor directives

- `@import imp1, imp2, ...` Add imports to this library
- `@declare` Declares a *declaration* code block
- `@init` Declares an *initialization* code block
- `@process` Declares a *processing* code block
- `@finalize` Declares a *finalization* code block

There can be several statements of each sort, several code block of the same kind will just be concatenated.
The first part of the library (without any statement) is the declaration.

The system is so done that libraries access, in their code, only variables they declare themselves or which are declared by imports they specify.

 ## Example

 ```rust
 // This library is used with some input/output declarations TS-side
 @declare
var<private> scaleFactor: f32;

@init
	scaleFactor = 2.0;
@process
	let scaledValue = input[thread.x] * scaleFactor;
@finalize
	output[thread.x] = scaledValue;
```

This library lets a code importing it change the `scaleFactor` variable without having to know about its existence, and even the `scaledInput` result before its writing into `output`.