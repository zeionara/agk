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
        lnEmb: [10, 10],
        lnBot: [2, 2],
        lnTop: [2, 2],
        nTrainEpochs: 100,
        learningRate: 0.001,
        trainBatchSize: 1024
)

//let dataset = SimpleDataset(trainPath: "train.txt", testPath: "test.txt")
//print(dataset.training)
