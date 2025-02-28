'use strict';
// https://github.com/y1j2x34/rollup-ts-karma-mocha-chai-boilerplate/blob/master/karma.base.conf.js
export default {
	basePath: '',

	frameworks: ['mocha', 'chai'],

	restartBrowserBetweenTests: false,

	files: [
		'tests/**/*.test.ts',
		{
			pattern: 'src/**/*.ts',
			served: true,
			included: false,
			watched: true
		}
	],

	mime: {
		'text/x-typescript': ['ts','tsx']
	},

	reporters: ['mocha'],

	port: 9876,

	colors: true,

	autoWatch: true,

	usePolling: true,

	atomic_save: false,
	customLaunchers: {
		HeadlessChrome: {
			base: 'ChromeHeadless',
			flags: [
				'--no-sandbox',
				'--headless',
				'--disable-translate',
				'--disable-extensions',
				'--enable-unsafe-webgpu',
				'--enable-webgpu-developer-features',
				'--enable-vulkan',
				'--enable-features=Vulkan',
			]
		},
		ChromeGPU: {
			base: 'Chrome',
			flags: [
				'--no-sandbox',
				'--disable-translate',
				'--disable-extensions',
				'--enable-unsafe-webgpu',
				'--enable-webgpu-developer-features',
				'--enable-vulkan',
				'--enable-features=Vulkan',
				'--remote-debugging-port=9222'
			]
		}
	},

	singleRun: false,

	concurrency: Infinity
};