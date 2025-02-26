'use strict';

import baseConfig from './karma.base.conf.js'
import rollupConfig from './rollup.tests.config.js'

export default function (config) {
    config.set(baseConfig);
    config.set({
        preprocessors: {
            'tests/**/*.test.ts': ['sourcemap', 'rollup']
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
            'karma-sourcemap-loader'
        ],
        singleRun: false,
    });
};