import TensorFlow

//testNeuMF(
//        size: [16, 32, 16, 8],
//        regs: [0.0, 0.0, 0.0, 0.0],
//        numLatentFeatures: 8,
//        matrixRegularization: 0.0,
//        learningRate: 0.001,
//        nTrainEpochs: 20
//)

//testDLRM(
//        nDense: 2,
//        mSpa: 2,
//        lnEmb: [100000, 100000],
//        lnBot: [200, 200],
//        lnTop: [200, 200],
//        nTrainEpochs: 1000,
//        learningRate: 0.07,
//        trainBatchSize: 1024,
//        nTestSamples: 4
//)

//let dataset = SimpleDataset(trainPath: "train.txt", testPath: "test.txt")
//print(dataset.training)
//print(Device.allDevices)
let device = Device.default
let dataset = KnowledgeGraphDataset(path: "train-ke-small.txt", device: device)
var model = GCN(dataset: dataset)
//print(model(Tensor<Int32>(dataset.frame.adjacencyTensor)))
//print(Tensor<Float>([[1, 2, 3], [4, 5, 6], [7, 8, 19]]).inverse)
let optimizer = Adam(for: model, learningRate: 0.01)
let trainer = GCNTrainer(nEpochs: 10, batchSize: 3)
trainer.train(dataset: dataset, model: &model, optimizer: optimizer, labels: Tensor<Float>([0.0, 1.0, 1.0]))
//print(dataset.frame.adjacencyTensor)
//let dataset = FlattenedKnowledgeGraphDataset(path: "train-ke-small.txt", device: device)
//print(dataset.negativeFrame)
//let batches = dataset.frame.batched(size: 3)
//print(batches)
//print(batches[2])
//var model = TransE(
//        embeddingDimensionality: 100,
//        dataset: dataset,
//        device: device
//)
//let optimizer = Adam(for: model, learningRate: 0.01)
//let trainer = Trainer(nEpochs: 100, batchSize: 3)
//let tens = Tensor<Float>([[0.1, 0.2], [1.0, 2.0], [3.0, 4.0]])
//let tester = Tester(batchSize: 3)
//trainer.train(dataset: dataset, model: &model, optimizer: optimizer)
//tester.test(dataset: dataset, model: model)
//print(dataset.frame.tensor.device)
//print(Device.allDevices)
//let score = model(dataset.normalizedFrame.tensor)
//print(score)
//print(score)
//print(dataset.headsUnique)
