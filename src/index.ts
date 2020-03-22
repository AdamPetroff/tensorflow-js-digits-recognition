import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"
import { Sequential } from "@tensorflow/tfjs"

import { MnistData } from "./data"

const IMAGE_WIDTH = 28
const IMAGE_HEIGHT = 28
const IMAGE_CHANNELS = 1

async function showExamples(data: MnistData) {
  const surface = tfvis.visor().surface({ name: "Examples", tab: "InputData" })

  const examples = data.nextTestBatch(20)
  const numExampleCount = examples.xs.shape[0]

  for (let i = 0; i < numExampleCount; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs.slice([i, 0], [1, examples.xs.shape[1]]).reshape([28, 28, 1])
    })

    const canvas = document.createElement("canvas")
    canvas.width = 28
    canvas.height = 28
    canvas.setAttribute("style", "margin: 4px;")
    await tf.browser.toPixels(imageTensor.as2D(IMAGE_WIDTH, IMAGE_HEIGHT), canvas)
    surface.drawArea.appendChild(canvas)
    imageTensor.dispose()
  }
}

function getModel() {
  const model = tf.sequential()

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    }),
  )

  // max-pooling
  model.add(tf.layers.maxPool2d({ poolSize: [2, 2], strides: [2, 2] }))

  // the stack again
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: "relu",
      kernelInitializer: "varianceScaling",
    }),
  )
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  model.add(tf.layers.flatten())

  // kernelInitializer is a function used to innitially select random weights
  // softmax for selecting one thing
  model.add(
    tf.layers.dense({ units: 10, kernelInitializer: "varianceScaling", activation: "softmax" }),
  )

  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  })

  return model
}

async function train(model: Sequential, data: MnistData) {
  const metrics = ["loss", "val_loss", "acc", "val_acc"]

  const BATCH_SIZE = 512
  const TRAIN_DATA_SIZE = 5500
  const TEST_DATA_SIZE = 1000

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE)

    return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE)

    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels]
  })

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 8,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks({ name: "Traing performance" }, metrics),
  })
}

const classNames = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

function doPrediction(model: Sequential, data: MnistData, testDataSize = 500) {
  const testData = data.nextTestBatch(testDataSize)
  const testXs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])
  const labels = testData.labels.argMax(-1)
  const predictions = model.predict(testXs)

  testXs.dispose()
  return [Array.isArray(predictions) ? predictions[0].argMax(-1) : predictions.argMax(-1), labels]
}

async function showAccuracy(model: Sequential, data: MnistData) {
  const [preds, labels] = doPrediction(model, data)
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels.as1D(), preds.as1D())
  tfvis.show.perClassAccuracy({ name: "Accuracy", tab: "Evaluation" }, classAccuracy, classNames)

  labels.dispose()
}

async function showConfusion(model: Sequential, data: MnistData) {
  const [preds, labels] = doPrediction(model, data)
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels.as1D(), preds.as1D())
  const container = { name: "Confusion Matrix", tab: "Evaluation" }
  tfvis.render.confusionMatrix(container, { values: confusionMatrix })

  labels.dispose()
}

async function run() {
  const data = new MnistData()
  await data.load()
  await showExamples(data)

  const model = getModel()
  await train(model, data)

  await showAccuracy(model, data)
  await showConfusion(model, data)
}

document.addEventListener("DOMContentLoaded", run)
