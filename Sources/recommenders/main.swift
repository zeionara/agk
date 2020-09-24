//testNeuMF(
//        size: [16, 32, 16, 8],
//        regs: [0.0, 0.0, 0.0, 0.0],
//        numLatentFeatures: 8,
//        matrixRegularization: 0.0,
//        learningRate: 0.001,
//        nTrainEpochs: 20
//)

testDLRM(
        nDense: 2,
        mSpa: 2,
        lnEmb: [100000, 100000],
        lnBot: [200, 200],
        lnTop: [200, 200],
        nTrainEpochs: 1000,
        learningRate: 0.07,
        trainBatchSize: 1024,
        nTestSamples: 4
)

//let dataset = SimpleDataset(trainPath: "train.txt", testPath: "test.txt")
//print(dataset.training)
