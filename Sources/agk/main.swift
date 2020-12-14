import TensorFlow
import ArgumentParser
import PythonKit
import Foundation
import Logging

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

let metrics: [Metric] = [
    MRR(n: 1), MRR(n: 2), MRR(n: 3), MRR(n: 4),
    Hits(n: 1), Hits(n: 2), Hits(n: 3), Hits(n: 4),
    MAP(n: 1), MAP(n: 2), MAP(n: 3), MAP(n: 4),
    NDCG(n: 1), NDCG(n: 2), NDCG(n: 3), NDCG(n: 4)
]

let classificationMetrics: [ClassificationMetric] = [
    Precision(0), Precision(0.2), Precision(0.4), Precision(0.6), Precision(0.8), Precision(1),
    Precision(0, reverse: true), // Precision(0.2, reverse: true), Precision(0.4, reverse: true), Precision(0.6, reverse: true), Precision(0.8, reverse: true), Precision(1, reverse: true),
    Recall(0), Recall(0.2), Recall(0.4), Recall(0.6), Recall(0.8), Recall(1),
    Recall(0, reverse: true), // Recall(0.2, reverse: true), Recall(0.4, reverse: true), Recall(0.6, reverse: true), Recall(0.8, reverse: true), Recall(1, reverse: true),
    F1Score(0), F1Score(0.2), F1Score(0.4), F1Score(0.6), F1Score(0.8), F1Score(1),
    F1Score(0, reverse: true), // F1Score(0.2, reverse: true), F1Score(0.4, reverse: true), F1Score(0.6, reverse: true), F1Score(0.8, reverse: true), F1Score(1, reverse: true),
    Accuracy(0), Accuracy(0.2), Accuracy(0.4), Accuracy(0.6), Accuracy(0.8), Accuracy(1),
    Accuracy(0, reverse: true) // Accuracy(0.2, reverse: true), Accuracy(0.4, reverse: true), Accuracy(0.6, reverse: true), Accuracy(0.8, reverse: true), Accuracy(1, reverse: true)
]

struct CrossValidate: ParsableCommand {

    private enum Model: String, ExpressibleByArgument {
        case transe
        case rotate
        case transd
        case gcn
        case vgae
        case conve
    }

    private enum GraphRepresentation: String, ExpressibleByArgument {
        case adjacencyMatrix
        case adjacencyPairsMatrix
    }

    private enum EmbeddingsDimensionalityReducer: String, ExpressibleByArgument {
        case sum
        case avg
    }

    @Option(name: .shortAndLong, help: "Model name which to use")
    private var model: Model

    @Option(default: .adjacencyMatrix, help: "Approach for representing graph as a tensor to use")
    private var graphRepresentation: GraphRepresentation

    @Option(default: .sum, help: "Approach for reducing embeddings dimensionality")
    private var embeddingsDimensionalityReducer: EmbeddingsDimensionalityReducer

    @Option(name: .shortAndLong, help: "Dataset filename (should be located in the 'data' folder)")
    private var datasetPath: String

    @Option(default: "humorous.txt", help: "Filename containing labels for graph nodes (should be located in the 'data' folder)")
    private var labelsPath: String

    @Option(name: .shortAndLong, default: 10, help: "Number of epochs to execute during model training")
    var nEpochs: Int

    @Option(default: 3, help: "Number of splits to perform for making the cross-validation")
    var nFolds: Int

    @Option(name: .shortAndLong, default: 20, help: "How many samples to put through the model at once")
    var batchSize: Int

    @Option(name: .shortAndLong, default: 10, help: "Size of vectors for embeddings generation")
    var embeddingDimensionality: Int

    @Option(name: .shortAndLong, default: 0.01, help: "How fast to tweak the weights")
    var learningRate: Float

    @Flag(name: .shortAndLong, help: "Perform computations on the gpu")
    var openke = false

    @Flag(name: .shortAndLong, help: "Use openke implementation")
    var gpu = false

    @Flag(help: "Do not retrain embeddings for classifier")
    var readEmbeddings = false

    @Option(default: 0.05, help: "How fast to tweak the weights")
    var classifierLearningRate: Float

    @Option(name: .shortAndLong, default: 10, help: "Number of epochs to execute during model training")
    var classifierNEpochs: Int

    mutating func run() throws {
        
        let embeddingsTensorDimensionalityReducers: [EmbeddingsDimensionalityReducer: (Tensor<Float>) -> Tensor<Float>]  = [
            .sum: { embeddings in
                embeddings.sum(alongAxes: [1]) // embeddings.mean(alongAxes: [1])
            },
            .avg : { embeddings in
                embeddings.mean(alongAxes: [1]) // embeddings.mean(alongAxes: [1])
            }
        ]

        let trainTensorPaths: [GraphRepresentation: KeyPath<KnowledgeGraphDataset<String, Int32>, Tensor<Float>>]  = [
            .adjacencyMatrix: \.tunedAdjecencyMatrixInverse,
            .adjacencyPairsMatrix : \.tunedAdjacencyPairsMatrixInverse
        ]

        let adjacencyTensorPaths: [GraphRepresentation: KeyPath<KnowledgeGraphDataset<String, Int32>, Tensor<Int8>>]  = [
            .adjacencyMatrix: \.frame.adjacencyTensor,
            .adjacencyPairsMatrix : \.adjacencyPairsTensor
        ]

        let device = gpu ? Device.defaultXLA : Device.default
        let dataset = KnowledgeGraphDataset<String, Int32>(path: datasetPath, device: device)
        let learningRate_ = learningRate
        let embeddingDimensionality_ = embeddingDimensionality
        var logger = Logger(label: "root")
        logger.logLevel = .trace

        let nEpochs_ = nEpochs
        let classifierLearningRate_ = classifierLearningRate
        let readEmbeddings_ = readEmbeddings
        let dataset_ = KnowledgeGraphDataset<String, Int32>(path: datasetPath, classes: labelsPath, device: device)
        let batchSize_ = batchSize

        let graphRepresentation_ = graphRepresentation

        let entityIndicesGetters: [GraphRepresentation: (LabelFrame<Int32>) -> Tensor<Int32>]  = [
            .adjacencyMatrix: { labels in
                labels.indices
            },
            .adjacencyPairsMatrix : { labels in
                dataset_.getAdjacencyPairsIndices(labels: labels)
            }
        ]
        
        if (model == .conve) {
            let dataset_ = KnowledgeGraphDataset<String, Int32>(path: datasetPath, classes: labelsPath, device: device)
            let modelSavingLock = NSLock()
            var embeddingsWereInitialized: Bool = false
            try GenericCVTester<DenseClassifier<ConvE<String, Int32>>, LabelFrame<Int32>, ClassificationTrainer>(nFolds: nFolds, nEpochs: classifierNEpochs, batchSize: batchSize).test(
                dataset: dataset_,
                metrics: classificationMetrics,
                enableParallelism: false
            ) { trainer, labels -> DenseClassifier<ConvE<String, Int32>> in
            
                // 1. Train the base model with node encodings if necessary
                
                let shouldInitializeEmbeddings = !embeddingsWereInitialized && !readEmbeddings_
                var model_ = !shouldInitializeEmbeddings ? try ConvE(dataset: dataset_, device: device) : ConvE(
                    embeddingDimensionality: embeddingDimensionality_,
                    stackedEmbeddingsWidth: 5,
                    stackedEmbeddingsHeight: 2,
                    filterWidth: 5,
                    filterHeight: 2,
                    nConvolutionalFilters: 2,
                    dataset: dataset_,
                    device: device
                )
                if shouldInitializeEmbeddings {
                    let trainer_ = ConvolutionAdjacencySequenceTrainer(nEpochs: nEpochs_, batchSize: batchSize_)
                    let optimizer_ = Adam<ConvE<String, Int32>>(for: model_, learningRate: learningRate_)
                    trainer_.train(
                        dataset: dataset_,
                        model: &model_,
                        optimizer: optimizer_
                    )
                    try model_.save()
                    modelSavingLock.unlock()
                    embeddingsWereInitialized = true
                }

                // 2. Train classification block

                var classifier = DenseClassifier(graphEmbedder: model_, device: device)
                var classification_optimizer_ = Adam<DenseClassifier>(for: classifier, learningRate: classifierLearningRate_)
                trainer.train(model: &classifier, optimizer: &classification_optimizer_, labels: dataset_.labelFrame!)
                return classifier
            } computeMetric: { model, metric, trainLabels, testLabels -> Float in
                let logits = model(entityIndicesGetters[graphRepresentation_]!(testLabels)).flattened()
                return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
            } getSamplesList: { dataset in
                dataset.labelFrame!
            }
        } else if (model == .vgae) {
            if openke {
                let model_name = model.rawValue
                throw ModelError.unsupportedModel(message: "Model \(model_name) is not implemented in the OpenKE library!")
            }

            //
            //
            //
            
            // var model_ = readEmbeddings ? try VGAE(dataset: dataset, device: device) :
            //     VGAE(embeddingDimensionality: embeddingDimensionality, dataset: dataset_, device: device, adjacencyTensorPath: adjacencyTensorPaths[graphRepresentation]!)
            // print(model_.entityEmbeddings.embeddings.shape)
            // if !readEmbeddings {
            //     let trainer_ = ConvolutionAdjacencyTrainer(nEpochs: nEpochs)
            //     var optimizer_ = Adam<VGAE<String, Int32>>(for: model_, learningRate: learningRate_)
            //     trainer_.train(dataset: dataset, model: &model_, optimizer: optimizer_,  trainTensorPath: trainTensorPaths[graphRepresentation]!, adjacencyTensorPath: adjacencyTensorPaths[graphRepresentation]!)
            //     try model_.save()
            // }
            // // print(dataset_.labelFrame!.indices)
            // // let indices = dataset_.getAdjacencyPairsIndices(labels: dataset_.labelFrame!)
            // var classifier = DenseClassifier(graphEmbedder: model_, device: device) { embeddings in
            //     embeddings.sum(alongAxes: [1]) // embeddings.mean(alongAxes: [1])
            // }
            // // print(classifier(indices).shape)
            // let classificationTrainer = ClassificationTrainer(nEpochs: classifierNEpochs, batchSize: batchSize)
            // var classification_optimizer_ = Adam<DenseClassifier<String, Int32>>(for: classifier, learningRate: classifierLearningRate_)
            // classificationTrainer.train(model: &classifier, optimizer: &classification_optimizer_, labels: dataset_.labelFrame!) { labels in
            //     dataset_.getAdjacencyPairsIndices(labels: labels)
            // }
            let embeddingsDimensionalityReducer_ = embeddingsDimensionalityReducer
            let graphRepresentation_ = graphRepresentation
            
            //
            //
            //

            let modelSavingLock = NSLock()
            var embeddingsWereInitialized: Bool = false
            try GenericCVTester<DenseClassifier<VGAE<String, Int32>>, LabelFrame<Int32>, ClassificationTrainer>(nFolds: nFolds, nEpochs: classifierNEpochs, batchSize: batchSize).test(
                dataset: dataset_,
                metrics: classificationMetrics,
                enableParallelism: false
            ) { trainer, labels -> DenseClassifier<VGAE<String, Int32>> in

                // 1. Train the base model with node encodings if necessary

                let shouldInitializeEmbeddings = !embeddingsWereInitialized && !readEmbeddings_
                var model_ = !shouldInitializeEmbeddings ? try VGAE(dataset: dataset, device: device) : VGAE(
                    embeddingDimensionality: embeddingDimensionality_,
                    dataset: dataset_,
                    device: device
                )
                if shouldInitializeEmbeddings {
                    let trainer_ = ConvolutionAdjacencyTrainer(nEpochs: nEpochs_)
                    let optimizer_ = Adam<VGAE<String, Int32>>(for: model_, learningRate: learningRate_)
                    trainer_.train(dataset: dataset, model: &model_, optimizer: optimizer_)
                    modelSavingLock.lock()
                    try model_.save()
                    modelSavingLock.unlock()
                    embeddingsWereInitialized = true
                }

                // 2. Train classification block

                var classifier = DenseClassifier(graphEmbedder: model_, device: device, reduceEmbeddingsTensorDimensionality: embeddingsTensorDimensionalityReducers[embeddingsDimensionalityReducer_]!) 
                var classification_optimizer_ = Adam<DenseClassifier>(for: classifier, learningRate: classifierLearningRate_)
                trainer.train(model: &classifier, optimizer: &classification_optimizer_, labels: labels, getEntityIndices: entityIndicesGetters[graphRepresentation_]!)
                return classifier
            }  computeMetric: { model, metric, trainLabels, testLabels -> Float in
                let logits = model(entityIndicesGetters[graphRepresentation_]!(testLabels)).flattened()
                return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
            } getSamplesList: { dataset in
                dataset.labelFrame!
            }
        } else if (model == .gcn) {
            if openke {
                let model_name = model.rawValue
                throw ModelError.unsupportedModel(message: "Model \(model_name) is not implemented in the OpenKE library!")
            }
            let dataset_ = KnowledgeGraphDataset<String, Int32>(path: datasetPath, classes: labelsPath, device: device)
            ClassificationCVTester<GCN<String, Int32>, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset_, metrics: classificationMetrics, enableParallelism: false) { frame, trainer, labels in
                var model_ = GCN(embeddingDimensionality: embeddingDimensionality_, dataset: dataset_, device: device, hiddenLayerSize: 100) // :TransE(embeddingDimensionality: embeddingDimensionality, dataset: dataset, device: device)
                var optimizer = Adam<GCN<String, Int32>>(for: model_, learningRate: learningRate_)
                trainer.train(dataset: dataset_, model: &model_, optimizer: &optimizer, labels: labels, frame: frame) // loss: computeSigmoidLoss
                return model_
            }
        } else if (model == .rotate) {
            if openke {
                let model_name = model.rawValue
                throw ModelError.unsupportedModel(message: "Rotate is not implemented in the OpenKE library!")
            } else {
                CVTester<RotatE<String, Int32>, LinearTrainer, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset, metrics: metrics) { trainFrame, trainer in
                    var model_ = RotatE(embeddingDimensionality: embeddingDimensionality_, dataset: dataset, device: device) // :TransE(embeddingDimensionality: embeddingDimensionality, dataset: dataset, device: device)
                    var optimizer = Adam<RotatE>(for: model_, learningRate: learningRate_)
                    trainer.train(frame: trainFrame, model: &model_, optimizer: &optimizer, loss: computeSigmoidLoss)
                    return model_
                }
            }
        } else if (model == .transe) {
            if openke {
                let model_name = model.rawValue
                CVTester<OpenKEModel, OpenKEModelTrainer, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset, metrics: metrics, enableParallelism: false) { trainFrame, trainer in
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
        } else if (model == .transd) {
            if openke {
                let model_name = model.rawValue
                CVTester<OpenKEModel, OpenKEModelTrainer, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset, metrics: metrics, enableParallelism: false) { trainFrame, trainer in
                    OpenKEModel(
                            configuration: trainer.train(model: model_name, frame: trainFrame, dataset: dataset)
                    )
                }
            } else {
                CVTester<TransD<String, Int32>, LinearTrainer, String>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(dataset: dataset, metrics: metrics) { trainFrame, trainer in
                    var model_ = TransD(embeddingDimensionality: embeddingDimensionality_, dataset: dataset, device: device) // :TransE(embeddingDimensionality: embeddingDimensionality, dataset: dataset, device: device)
                    var optimizer = Adam<TransD>(for: model_, learningRate: learningRate_)
                    trainer.train(frame: trainFrame, model: &model_, optimizer: &optimizer, loss: computeSigmoidLoss)
                    return model_
                }
            }
        } else {
            throw ModelError.unsupportedModel(message: "Model \(model) is not supported yet!")
        }
    }
}

public struct ReportEntry {
    public let header: String
    public var metrics: [String: Float]
}

struct RestructureReport: ParsableCommand {

    @Argument(help: "Path to file containing input report")
    private var inputPath: String

    @Argument(help: "Path to output file to save generated table")
    private var outputPath: String

    @Option(name: .shortAndLong, default: 3, help: "Precision with which values will be printed")
    var nDecimalPlaces: Int

    private func readData(path: String) throws -> [ReportEntry] {
        func skipEmptyLines() {
            while lines[offset] == "" && offset < lines.count - 1 {
                offset += 1
            }
        }

        var reportEntries = [ReportEntry]()
        let lines = try readLines(path: path)
        var offset = 0

        do {
            while offset < lines.count {
//                print("a")
                skipEmptyLines()
//                print("b")
                let header = lines[offset]
//                print("c")
                offset += 1
//                print("d")
//                print(header)
//                print(offset)
                if offset >= lines.count {
                    break
                }
                skipEmptyLines()
//                print("e")
                var metrics = [String: Float]()
//                print("f")
                while lines[offset] != "" {
                    let metricWithValue = lines[offset].components(separatedBy: ": ")
                    metrics[metricWithValue[0]] = metricWithValue[1] != "nan" ? Float(metricWithValue[1])! : Float.nan
                    // print(metrics[metricWithValue[0]])
                    // print(type(of: metrics[metricWithValue[0]]))
                    // print(metrics[metricWithValue[0]]!.isNaN)
                    offset += 1
                }
//                print(offset)
                reportEntries.append(ReportEntry(header: header, metrics: metrics))
            }
        } catch {
        }
        return reportEntries
    }

    private func writeData(path: String, data: [ReportEntry]) throws {
        let metric_keys = data[0].metrics.keys.sorted()
        let model_keys = data.map {
            $0.header
        }
        var lines: [String] = ["metric\t\(model_keys.joined(separator: "\t"))"] + metric_keys.map { metric in
            "\(metric)\t\(model_keys.map{data[$0]!.metrics[metric]!.isNaN ? "-" : String(format: "%.\(nDecimalPlaces)f", data[$0]!.metrics[metric]!)}.joined(separator: "\t"))"
        }
        try writeLines(path: path, lines: lines)
    }

    mutating func run() throws {
        let data = try readData(path: inputPath)
        try writeData(path: outputPath, data: data)
    }
}

struct Agk: ParsableCommand {
    static var configuration = CommandConfiguration(
            abstract: "A tool for automating operation on the knowledge graph models",
            subcommands: [CrossValidate.self, RestructureReport.self], // TrainExternally.self
            defaultSubcommand: CrossValidate.self
    )
}

Agk.main()
