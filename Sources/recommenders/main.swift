import TensorFlow
import ArgumentParser
import PythonKit

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
//let device = Device.default
//let dataset = KnowledgeGraphDataset<String, Int32>(path: "truncated-dataset-normalized.txt", device: device)
//for (i, triple) in dataset.negativeFrame.data.enumerated() {
//    print(triple)
//    if (i > 3) {
//        break
//    }
//}
//print(dataset.normalizedFrame)
//let chunks = dataset.normalizedFrame.split(nChunks: 2)
//let props = chunks[0].split(proportions: [0.3, 0.7])
//print(props[1].data.count)
//var model = TransE(dataset: dataset)
//var model = ConvE(embeddingDimensionality: 50, stackedEmbeddingsWidth: 10, stackedEmbeddingsHeight: 5, filterWidth: 3, filterHeight: 3, dataset: dataset)
//print(sigmoidCrossEntropy(logits: model(Tensor<Int32>(dataset.normalizedFrame.tensor)), labels: dataset.normalizedFrame.adjacencySequence))
//print(model(Tensor<Int32>(dataset.normalizedFrame.tensor)))
//print(Tensor<Float>([[1, 2, 3], [4, 5, 6], [7, 8, 19]]).inverse)
//var optimizer = Adam(for: model, learningRate: 0.01)
//let trainer = ConvolutionAdjacencySequenceTrainer(nEpochs: 50, batchSize: 3)
//let trainer = LinearTrainer(nEpochs: 1, batchSize: 256)
//trainer.train(frame: dataset.normalizedFrame, model: &model, optimizer: &optimizer, loss: computeSigmoidLoss)

//trainer.train(dataset: dataset, model: &model, optimizer: optimizer)
// CV pipeline
//let array = [0, 1, 2, 3]
//print(array.getCombinations(k: 1))
//let tester = CVTester<RotatE<String, Int32>, Adam<RotatE>, LinearTrainer, String>(nFolds: 10, nEpochs: 100, batchSize: 256).test(dataset: dataset, metrics: [
//    MRR(n: 1), MRR(n: 2), MRR(n: 3), MRR(n: 4),
//    Hits(n: 1), Hits(n: 2), Hits(n: 3), Hits(n: 4),
//    MAP(n: 1), MAP(n: 2), MAP(n: 3), MAP(n: 4),
//    NDCG(n: 1), NDCG(n: 2), NDCG(n: 3), NDCG(n: 4)
//])
//{ trainFrame, trainer in
//    var model = RotatE(embeddingDimensionality: 100, dataset: dataset, device: Device.default)
//    var optimizer = Adam<RotatE>(for: model, learningRate: 0.01)
//    trainer.train(frame: trainFrame, model: &model, optimizer: &optimizer, loss: computeSigmoidLoss)
//    return model
//}
//var scores: [Float] = []
//let metric = RandomMetric(k: 2.2)
//for (trainFrame, testFrame) in dataset.normalizedFrame.cv(nFolds: 4) {
//    var model = RotatE(embeddingDimensionality: 100, dataset: dataset)
//    let optimizer = Adam(for: model, learningRate: 0.01)
//    trainer.train(frame: trainFrame, model: &model, optimizer: optimizer, loss: computeSigmoidLoss)
//    scores.append(metric.compute(model: model, trainFrame: trainFrame, testFrame: testFrame))
//}
//print("\(metric.name): \(metric.aggregate(scores: scores))")
//let tensor = Tensor<Float>([0.1, 0.2, 0.3])
//print(tensor.gathering(atIndices: Tensor<Int32>([0, 2])))
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

enum ModelError: Error {
    case unsupportedModel(message: String)
}

let metrics: [LinearMetric] = [
    MRR(n: 1), MRR(n: 2), MRR(n: 3), MRR(n: 4),
    Hits(n: 1), Hits(n: 2), Hits(n: 3), Hits(n: 4),
    MAP(n: 1), MAP(n: 2), MAP(n: 3), MAP(n: 4),
    NDCG(n: 1), NDCG(n: 2), NDCG(n: 3), NDCG(n: 4)
]

struct CrossValidate: ParsableCommand {

    private enum Model: String, ExpressibleByArgument {
        case transe
        case rotate
        case transd
    }

    @Option(name: .shortAndLong, help: "Model name which to use")
    private var model: Model

    @Option(name: .shortAndLong, help: "Dataset filename (should be located in the 'data' folder)")
    private var datasetPath: String

    @Option(name: .shortAndLong, default: 100, help: "Number of epochs to execute during model training")
    var nEpochs: Int

    @Option(default: 10, help: "Number of splits to perform for making the cross-validation")
    var nFolds: Int

    @Option(name: .shortAndLong, default: 256, help: "How many samples to put through the model at once")
    var batchSize: Int

    @Option(name: .shortAndLong, default: 100, help: "Size of vectors for embeddings generation")
    var embeddingDimensionality: Int

    @Option(name: .shortAndLong, default: 0.01, help: "How fast to tweak the weights")
    var learningRate: Float

    @Flag(name: .shortAndLong, help: "Perform computations on the gpu")
    var openke = false

    @Flag(name: .shortAndLong, help: "Use openke implementation")
    var gpu = false

    mutating func run() throws {
        let device = gpu ? Device.defaultXLA : Device.default
        let dataset = KnowledgeGraphDataset<String, Int32>(path: datasetPath, device: device)
        var learningRate_ = learningRate
        var embeddingDimensionality_ = embeddingDimensionality

        if (model == .rotate) {
            CVTester<RotatE<String, Int32>, LinearTrainer, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset, metrics: metrics) { trainFrame, trainer in
                var model_ = RotatE(embeddingDimensionality: embeddingDimensionality_, dataset: dataset, device: device) // :TransE(embeddingDimensionality: embeddingDimensionality, dataset: dataset, device: device)
                var optimizer = Adam<RotatE>(for: model_, learningRate: learningRate_)
                trainer.train(frame: trainFrame, model: &model_, optimizer: &optimizer, loss: computeSigmoidLoss)
                return model_
            }
        } else if (model == .transe) {
            if openke {
                let model_name = model.rawValue
                CVTester<OpenKEModel, OpenKEModelTrainer, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset, metrics: metrics) { trainFrame, trainer in
                    OpenKEModel(
                            configuration: trainer.train(model: model_name, frame: trainFrame, dataset: dataset)
                    )
                }
            } else {
                CVTester<TransE<String, Int32>, LinearTrainer, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset, metrics: metrics) { trainFrame, trainer in
                    var model_ = TransE(embeddingDimensionality: embeddingDimensionality_, dataset: dataset, device: device)
                    var optimizer = Adam<TransE>(for: model_, learningRate: learningRate_)
                    trainer.train(frame: trainFrame, model: &model_, optimizer: &optimizer)
                    return model_
                }
            }
        } else {
            throw ModelError.unsupportedModel(message: "Model \(model) is not supported yet!")
        }
    }
}

struct Agk: ParsableCommand {
    static var configuration = CommandConfiguration(
            abstract: "A tool for automating operation on the knowledge graph models",
            subcommands: [CrossValidate.self], // TrainExternally.self
            defaultSubcommand: CrossValidate.self
    )
}

Agk.main()
