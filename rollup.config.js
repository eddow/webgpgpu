import { rm } from 'node:fs/promises'
import commonjs from '@rollup/plugin-commonjs'
import resolve from '@rollup/plugin-node-resolve'
import strip from '@rollup/plugin-strip'
import typescript from '@rollup/plugin-typescript'
import pluginDts from 'rollup-plugin-dts'

// clean out the destination folder
await rm('dist', { recursive: true, force: true })
const banner = `/*
	webgpgpu.ts - https://github.com/eddow/webgpgpu
*/`
const input = { client: 'src/client/index.ts', server: 'src/server/index.ts' }
const plugins = [
	resolve(),
	commonjs(),
	typescript({
		tsconfig: './src/tsconfig.json',
	}),
]
const external = (id) => id.includes('node_modules')

export default [
	{
		input,
		output: {
			banner,
			dir: 'dist',
		},
		plugins: [...plugins, pluginDts()],
		external,
	},
	{
		input,
		output: [
			{
				banner,
				dir: 'dist',
				entryFileNames: '[name].js',
				sourcemap: true,
				format: 'esm',
			},
			{
				banner,
				dir: 'dist',
				entryFileNames: '[name].cjs',
				chunkFileNames: '[name].cjs',
				sourcemap: true,
				format: 'cjs',
				exports: 'named',
			},
		],
		plugins: [
			...plugins,
			strip({
				exclude: ['src/log.ts'],
			}),
		],
		external,
	},
]
