import { Indexable } from '../src/hacks'

class MyClass extends Indexable<number> {
	getAtIndex(index: number): number {
		return index + 1
	}
}

const myInstance = new MyClass()
const t = myInstance[3]
console.log(myInstance[5])
myInstance[4] = 8
console.log(myInstance[4])
