const path = require("path")
module.exports = {
  mode: "development",
  devtool: "inline-source-map",
  context: __dirname,
  entry: ["@babel/polyfill", "./src/index.ts"],
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "index.js",
  },
  resolve: {
    mainFields: ["browser", "main", "module"],
    extensions: [".ts", ".tsx", ".js", ".jsx", ".mjs"],
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx|ts|tsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
        },
      },
      {
        test: /\.js$/,
        use: ["source-map-loader"],
        enforce: "pre",
      },
    ],
  },
  devServer: {
    contentBase: "./dist",
  },
  node: {
    fs: "empty",
  },
}
