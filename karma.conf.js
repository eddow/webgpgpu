import rollupConfig from './rollup.tests.config.js'

export default function (config) {
	config.set({
		basePath: '',

		frameworks: ['mocha', 'chai'],

		restartBrowserBetweenTests: false,

		files: [
			'tests/**/*.test.ts',
			{
				pattern: 'src/**/*.ts',
				served: true,
				included: false,
				watched: true,
			},
		],

		mime: {
			'text/x-typescript': ['ts', 'tsx'],
		},

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
				],
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
					'--remote-debugging-port=9222',
				],
			},
		},

		singleRun: false,

		concurrency: Number.POSITIVE_INFINITY,
		preprocessors: {
			'tests/**/*.test.ts': ['sourcemap', 'rollup'],
		},
		rollupPreprocessor: rollupConfig,
		reporters: ['progress', 'mocha'],

		logLevel: config.LOG_WARN,

		plugins: [
			'karma-chrome-launcher',
			'karma-mocha',
			'karma-chai',
			'karma-mocha-reporter',
			'karma-rollup-preprocessor',
			'karma-sourcemap-loader',
		],
	})
}
