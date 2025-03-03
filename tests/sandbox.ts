import { f32 } from 'webgpgpu'
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
