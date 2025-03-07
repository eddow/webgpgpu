/*import { f32 } from 'webgpgpu'
const inferences = { x: undefined, y: undefined }
const type = f32.array('x', 'y')
const ab = type.toArrayBuffer(
	[
		[1, 2, 3, 4, 5],
		[6, 7, 8, 9, 10],
	],
	inferences
)
const br = type.readArrayBuffer(ab, inferences)
const rv = Array.from(br)
// TODO: display BufferReader in node (MaximumCallStack exception)
console.log(rv)
*/
/*
function Callable<Args extends any[], Rv>(fct: (this: any, ...args: Args) => Rv) {
	class Callable {}
	Object.setPrototypeOf(
		Callable.prototype,
		new Proxy(Callable.prototype, {
			apply(target, thisArg, argArray) {
				//debugger
			},
		})
	)
	return Callable // as typeof Callable & ((...args: Args) => Rv)
}

const C = Callable((a: string) => `Hello ${a}`)
class MyCallable extends C implements InstanceType<typeof C> {
	constructor(public b: string) {
		super()
	}
}

const c = new MyCallable('Dugenou') as unknown as (a: string) => string
console.log(c('Benjamin'))
*/
