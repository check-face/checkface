const realFs = require('fs')
const gracefulFs = require('graceful-fs')
gracefulFs.gracefulify(realFs) //patch graceful filesystem


var path = require('path');
var HtmlWebpackPlugin = require('html-webpack-plugin');
var CopyWebpackPlugin = require('copy-webpack-plugin');
var MiniCssExtractPlugin = require('mini-css-extract-plugin');
const { CleanWebpackPlugin } = require('clean-webpack-plugin')
const ReactRefreshWebpackPlugin = require('@pmmmwh/react-refresh-webpack-plugin');;


// If we're running the webpack-dev-server, assume we're in development mode
var isSelfHost = process.env.FACEMORPH_SELF_HOST === '1';
var isProduction = !process.argv.find(v => v.indexOf('webpack-dev-server') !== -1);
console.log('Bundling for ' + (isProduction ? 'production' : 'development') + '...');

var CONFIG = {
    // The tags to include the generated JS and CSS will be automatically injected in the HTML template
    // See https://github.com/jantimon/html-webpack-plugin
    indexHtmlTemplate: './src/index.html',
    fsharpEntry: './src/Index.fs.js',
    cssEntry: './src/style.scss',
    outputDir: isSelfHost ? './deploy-self-host' : './deploy',
    assetsDir: './src/public',
    devServerPort: 8100,

    // Use babel-preset-env to generate JS compatible with most-used browsers.
    // More info at https://babeljs.io/docs/en/next/babel-preset-env.html
    babel: {
        plugins: [(!isProduction) && require.resolve('react-refresh/babel')].filter(Boolean),
        presets: [ "@babel/preset-env", "@babel/preset-react" ]
    }
}


// The HtmlWebpackPlugin allows us to use a template for the index.html page
// and automatically injects <script> or <link> tags for generated bundles.
var commonPlugins = [
    new (require('webpack').DefinePlugin)({
        'process.env.FACEMORPH_SELF_HOST': JSON.stringify(isSelfHost ? '1' : '0'),
        'process.env.FACEMORPH_TRIAL': JSON.stringify(isSelfHost ? '0' : (process.env.FACEMORPH_TRIAL || '0')),
        'process.env.FACEMORPH_REVIEW': JSON.stringify(isSelfHost ? '0' : (process.env.FACEMORPH_REVIEW || '0')),
        'process.env.FACEMORPH_TRIAL_URL': JSON.stringify(process.env.FACEMORPH_TRIAL_URL || '')
    }),
    new HtmlWebpackPlugin({
        filename: 'index.html',
        selfHost: isSelfHost,
        template: resolve(CONFIG.indexHtmlTemplate)
    })
];

let client =

{
    // In development, split the JavaScript and CSS files in order to
    // have a faster HMR support. In production bundle styles together
    // with the code because the MiniCssExtractPlugin will extract the
    // CSS in a separate files.
    entry: isProduction ? {
        app: [resolve(CONFIG.fsharpEntry), resolve(CONFIG.cssEntry)]
    } : {
            app: [resolve(CONFIG.fsharpEntry)],
            style: [resolve(CONFIG.cssEntry)]
        },
    // Add a hash to the output file name in production
    // to prevent browser caching if code changes
    output: {
        path: resolve(CONFIG.outputDir),
        filename: isProduction ? '[name].[hash].js' : '[name].js'
    },
    mode: isProduction ? 'production' : 'development',
    devtool: isProduction ? 'source-map' : 'eval-source-map',
    optimization: {
        runtimeChunk: "single",
        moduleIds: 'deterministic',
        // Split the code coming from npm packages into a different file.
        // 3rd party dependencies change less often, let the browser cache them.
        splitChunks: {
            cacheGroups: {
                commons: {
                    test: /node_modules/,
                    name: "vendors",
                    chunks: "all"
                }
            }
        },
        minimize: isProduction
    },
    // Besides the HtmlPlugin, we use the following plugins:
    // PRODUCTION
    //      - MiniCssExtractPlugin: Extracts CSS from bundle to a different file
    //          To minify CSS, see https://github.com/webpack-contrib/mini-css-extract-plugin#minimizing-for-production
    //      - CopyWebpackPlugin: Copies static assets to output directory
    // DEVELOPMENT
    //      - HotModuleReplacementPlugin: Enables hot reloading when code changes without refreshing
    plugins: isProduction ?
        commonPlugins.concat([
            new MiniCssExtractPlugin({ filename: 'style.[hash].css' }),
            new CopyWebpackPlugin({ patterns: [
                { from: resolve(CONFIG.assetsDir) },
                ...(isSelfHost ? [] : [
                    { from: resolve("vercel.json") },
                    { from: resolve("vercel.package.json"), to: "package.json" },
                ]),
            ]}),
            new CleanWebpackPlugin({
                cleanOnceBeforeBuildPatterns: ['**/*', '!api/**'],
            }),
        ])
        : commonPlugins.concat([
            new ReactRefreshWebpackPlugin(),
        ]),
    resolve: {
        // See https://github.com/fable-compiler/Fable/issues/1490
        symlinks: false,
        modules: [resolve("./node_modules")],
        alias: {
            // Some old libraries still use an old specific version of core-js
            // Redirect the imports of these libraries to the newer core-js
            'core-js/es6': 'core-js/es'
        }
    },
    // Configuration for webpack-dev-server
    devServer: {
        static: {
            directory: resolve(CONFIG.assetsDir),
            publicPath: '/',
        },
        host: '0.0.0.0',
        port: CONFIG.devServerPort,
        hot: true,
        historyApiFallback: true
    },
    // - babel-loader: transforms JS to old syntax (compatible with old browsers)
    // - sass-loaders: transforms SASS/SCSS into JS
    // - file-loader: Moves files referenced in the code (fonts, images) into output folder
    module: {
        rules: [
            {
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                use: {
                    loader: 'babel-loader',
                    options: CONFIG.babel
                },
            },
            {
                test: /\.(sass|scss|css)$/,
                use: [
                    isProduction
                        ? MiniCssExtractPlugin.loader
                        : 'style-loader',
                    'css-loader',
                    {
                        loader: 'resolve-url-loader',
                    },
                    {
                      loader: 'sass-loader',
                      options: { implementation: require('sass') }
                    }
                ],
            },
            {
                test: /\.md$/,
                use: [
                    {
                        loader: 'html-loader'
                    },
                    {
                        loader: 'markdown-loader',
                        options: {

                        }
                    }
                ]
            },
            {
                test: /\.svg$/,
                issuer: /\.[jt]sx?$/,
                use: [
                    {
                        loader: '@svgr/webpack',
                        options: {
                            svgo: false
                        }
                    }
                ]
            },
            {
                test: /\.(png|jpg|jpeg|gif|woff|woff2|ttf|eot)(\?.*)?$/,
                use: ['file-loader']
            }
        ]
    },
};

let server =
{
    entry: {server: ['./Server/Index.fs.js']},
    target: 'node',
    output: {
        path: resolve("./deploy/api"),
        filename: '[name].js',
        libraryTarget: "commonjs2"
    },
    mode: isProduction ? 'production' : 'development',
    devtool: isProduction ? 'source-map' : 'eval-source-map',
    // See https://github.com/fable-compiler/Fable/issues/1490
    resolve: client.resolve,
    plugins: [
        new (require("webpack").DefinePlugin)({"process.env.FACEMORPH_SELF_HOST": JSON.stringify("0"), "process.env.FACEMORPH_TRIAL": JSON.stringify(process.env.FACEMORPH_TRIAL || "0"), "process.env.FACEMORPH_REVIEW": JSON.stringify(process.env.FACEMORPH_REVIEW || "0"), "process.env.FACEMORPH_TRIAL_URL": JSON.stringify(process.env.FACEMORPH_TRIAL_URL || "")}),
        new CleanWebpackPlugin()
    ],
    module: client.module,
    optimization: {
        minimize: isProduction
    },
    node: {
        __dirname: false,
        __filename: false,
    }
}

function resolve(filePath) {
    return path.isAbsolute(filePath) ? filePath : path.join(__dirname, filePath);
}

if(isProduction && !isSelfHost) {
    module.exports = [ client, server ]
}
else {
    module.exports = [ client ]
}
