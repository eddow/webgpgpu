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

import { call } from 'mocha'
import { _ } from '../lib/outputs-BYoh-eu2'

abstract class Callable<Args extends any[], Rv> {
	apply(_thisArg: any, args: Args): Rv {
		return this.call(_thisArg, ...args)
	}
	abstract call(_: any, ...args: Args): Rv
}
Object.setPrototypeOf(
	Callable.prototype,
	new Proxy(Callable.prototype, {
		apply(target, thisArg, argArray) {
			return target.call(thisArg, ...argArray)
		},
	})
)

type ICallable<Args extends any[], Rv> = Callable<Args, Rv> & ((...args: Args) => Rv)

class MyCallable extends ICallable<[string], string> {
	constructor(public b: string) {
		super()
	}
	call(_, a: string) {
		return `Bonjour ${a} ${this.b}`
	}
}

const c = new MyCallable('Dugenou')
console.log(c('Benjamin'))
