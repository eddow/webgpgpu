import createWebGpGpu, { f32, u32, type WebGpGpu } from '../src/server'

let webGpGpu: WebGpGpu

function cpu(n: number) {
	const result = new Float32Array(n)
	for (let i = 0; i < n; i++) {
		const mod = i % 453
		result[i] = mod * mod
	}
	return Promise.resolve(result)
}
async function benchmark(fct: (n: number) => Promise<unknown>, n: number) {
	performance.mark(`start ${fct.name} ${n}`)
	await fct(1 << n)
	performance.mark(`end ${fct.name} ${n}`)
	return performance.measure(fct.name, `start ${fct.name} ${n}`, `end ${fct.name} ${n}`).duration
}
async function main() {
	webGpGpu = await createWebGpGpu()
	const gpuSquaresKernel = webGpGpu.output({ output: u32.array('threads.x') }).kernel(/*wgsl*/ `
let modX = thread.x % 453;
output[thread.x] = modX * modX;
`)
	function gpu(n: number) {
		return gpuSquaresKernel({}, { 'threads.x': n })
	}
	/*await gpu(65536)*/
	console.log(webGpGpu.device.limits.maxBufferSize / 4) //*/
	for (let exp = 2; exp <= 5; exp++) {
		console.log('---')
		console.log(exp, ':', await benchmark(gpu, exp), '|', await benchmark(cpu, exp))
	}
	await new Promise((resolve) => setTimeout(resolve, 2000))
	webGpGpu.dispose()
}

main()
