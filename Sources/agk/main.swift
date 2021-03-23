import TensorFlow
import ArgumentParser
import PythonKit
import Foundation
import Logging

enum ModelError: Error {
    case unsupportedModel(message: String)
}

enum LossError: Error {
    case unsupportedLoss(message: String)
}

enum TaskError: Error {
    case unsupportedTask(message: String)
}

enum OptimizerError: Error {
    case unsupportedOptimizer(message: String)
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

var trainLogger = Logger(label: "trainer")
trainLogger.logLevel = .trace

func += <K, V> (left: inout [K:V], right: [K:V]) { 
    for (k, v) in right { 
        left[k] = v
    } 
}

func trainLinearModel<Model, OptimizerType>(
    trainer: LinearTrainer, trainFrame: TripleFrame<Int32>, model: inout Model, optimizer: inout OptimizerType,
    modelName: String = "unknown", loss: CrossValidate.LinearModelLoss, lossName: String = "unknown", margin: Float
) throws where Model: LinearGraphModel, OptimizerType: Optimizer, OptimizerType.Model == Model, Model.Scalar == Int32 {
    if loss == .sum {
        trainer.train(
            frame: trainFrame,
            model: &model,
            optimizer: &optimizer,
            logger: trainLogger,
            margin: margin,
            loss: computeSumLoss
        )
    } else if loss == .sigmoid {
        trainer.train(
            frame: trainFrame,
            model: &model,
            optimizer: &optimizer,
            logger: trainLogger,
            margin: margin,
            loss: computeSigmoidLoss
        )
    } else {
        throw LossError.unsupportedLoss(message: "Loss \(lossName) is not supported for model \(modelName)")
    }
}

struct CrossValidate: ParsableCommand, Encodable {

    enum CodingKeys: String, CodingKey {
        case model
        case datasetPath // = "dataset"
        case graphRepresentation // = "graph-representation"
        case embeddingsDimensionalityReducer //  = "embedding-dimensionality-reducer"
        case labelsPath // = "labels"
        case textsPath // = "texts"
        case nEpochs // = "n-epochs"
        case nFolds // = "n-folds"
        case batchSize // = "batch-size"
        case embeddingDimensionality // = "embedding-dimensionality"
        case learningRate // = "learning-rate"
        case openke // = "open-ke-implementation"
        case gpu // = "gpu-enabled"
        case readEmbeddings // = "precomputed-graph-embeddings"
        case classifierLearningRate // = "classifier-learning-rate"
        case classifierNEpochs // = "classifier-n-epochs"
        case languageModelName // = "language-model"
        case stackedEmbeddingsWidth // = "stacked-embeddings-width"
        case stackedEmbeddingsHeight // = "stacked-embeddings-height"
        case convolutionFilterWidth // = "convolution-filter-width"
        case convolutionFilterHeight // = "convolution-filter-height"
        case nConvolutionFilters // = "convolution-filters"
        case hiddenLayerSize // = "hidden-layer-size"
        case task // = "task"
        case margin // = "margin"
        case linearModelLoss // = "loss"
        case optimizer // = "optimizer"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(model.rawValue, forKey: .model)
        try container.encode(datasetPath, forKey: .datasetPath)
        try container.encode(graphRepresentation.rawValue, forKey: .graphRepresentation)
        try container.encode(embeddingsDimensionalityReducer.rawValue, forKey: .embeddingsDimensionalityReducer)
        try container.encode(labelsPath, forKey: .labelsPath)
        try container.encode(textsPath, forKey: .textsPath)
        try container.encode(nEpochs, forKey: .nEpochs)
        try container.encode(nFolds, forKey: .nFolds)
        try container.encode(batchSize, forKey: .batchSize)
        try container.encode(embeddingDimensionality, forKey: .embeddingDimensionality)
        try container.encode(learningRate, forKey: .learningRate)
        try container.encode(openke, forKey: .openke)
        try container.encode(gpu, forKey: .gpu)
        try container.encode(readEmbeddings, forKey: .readEmbeddings)
        try container.encode(classifierLearningRate, forKey: .classifierLearningRate)
        try container.encode(classifierNEpochs, forKey: .classifierNEpochs)
        try container.encode(languageModelName ?? "none", forKey: .languageModelName)
        try container.encode(stackedEmbeddingsWidth, forKey: .stackedEmbeddingsWidth)
        try container.encode(stackedEmbeddingsHeight, forKey: .stackedEmbeddingsHeight)
        try container.encode(convolutionFilterWidth, forKey: .convolutionFilterWidth)
        try container.encode(convolutionFilterHeight, forKey: .convolutionFilterHeight)
        try container.encode(nConvolutionFilters, forKey: .nConvolutionFilters)
        try container.encode(hiddenLayerSize, forKey: .hiddenLayerSize)
        try container.encode(task.rawValue, forKey: .task)
        try container.encode(margin, forKey: .margin)
        try container.encode(linearModelLoss.rawValue, forKey: .linearModelLoss)
        try container.encode(optimizer.rawValue, forKey: .optimizer)
    }

    private enum Model: String, ExpressibleByArgument {
        case transe
        case rotate
        case transd
        case gcn
        case vgae
        case conve
        case qrescal
    }

    private enum GraphRepresentation: String, ExpressibleByArgument {
        case adjacencyMatrix
        case adjacencyPairsMatrix
    }

    private enum EmbeddingsDimensionalityReducer: String, ExpressibleByArgument {
        case sum
        case avg
    }

    private enum Task: String, ExpressibleByArgument {
        case linkPrediction
        case classification
        case defaultTask
    }

    public enum LinearModelLoss: String, ExpressibleByArgument {
        case sum
        case sigmoid
        case defaultLoss
    }

    public enum Optimizer: String, ExpressibleByArgument {
        case sgd
        case adam
        case defaultOptimizer
    }

    @Option(name: .shortAndLong, help: "Model name which to use")
    private var model: Model

    @Option(default: .adjacencyMatrix, help: "Approach for representing graph as a tensor to use")
    private var graphRepresentation: GraphRepresentation

    @Option(default: .sum, help: "Approach for reducing embeddings dimensionality")
    private var embeddingsDimensionalityReducer: EmbeddingsDimensionalityReducer

    @Option(name: .shortAndLong, help: "Dataset filename (should be located in the 'data' folder)")
    private var datasetPath: String

    @Option(help: "Filename containing labels for graph nodes (should be located in the 'data' folder)")
    private var labelsPath: String = "humorous.txt"

    @Option(help: "Filename containing texts for graph nodes (should be located in the 'data' folder)")
    private var textsPath: String = "deduplicated-dataset-texts.txt"

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

    @Option(default: .none, help: "Name of the cached language model to apply inside dense-classification module")
    var languageModelName: String?

    @Option(default: 5, help: "Width of matrix being a result of reshaping an entity embedding")
    var stackedEmbeddingsWidth: Int

    @Option(default: 2, help: "Height of matrix being a result of reshaping an entity embedding")
    var stackedEmbeddingsHeight: Int

    @Option(default: 5, help: "Width of a convolution filter")
    var convolutionFilterWidth: Int

    @Option(default: 1, help: "Height of a convolution filter")
    var convolutionFilterHeight: Int

    @Option(default: 2, help: "Number of convolution filters")
    var nConvolutionFilters: Int

    @Option(default: 100, help: "Number of nodes in the hidden layer")
    var hiddenLayerSize: Int

    @Option(name: .shortAndLong, default: .defaultTask, help: "Name of task for performing cross validation testing")
    private var task: Task

    @Option(help: "Target distance between embeddings of the positive and negative triples")
    var margin: Float = 2.0

    @Option(help: "Name of loss for using during linear model training")
    private var linearModelLoss: LinearModelLoss = .defaultLoss

    @Option(help: "Type of optimizer to use for model training")
    private var optimizer: Optimizer = .defaultOptimizer

    public init(model: String, dataset: String) {
        self.model = Model.init(rawValue: model)!
        datasetPath = dataset
    }

    public init() { }

    mutating func run(_ result: inout [String: Any]) throws {

        func trainLinearModelUsingOptimizer<Model>(model: inout Model, trainer: LinearTrainer, trainFrame: TripleFrame<Int32>) throws
        where Model: LinearGraphModel, Model.TangentVector.VectorSpaceScalar == Float, Model.Scalar == Int32 {
            if optimizer_ == .adam {
                var optimizer = Adam<Model>(for: model, learningRate: learningRate_)
                try trainLinearModel(
                    trainer: trainer, trainFrame: trainFrame, model: &model, optimizer: &optimizer,
                    modelName: modelName, loss: linearModelLoss_, lossName: linearModelLossName, margin: margin_
                )
            } else if optimizer_ == .sgd {
                var optimizer = SGD<Model>(for: model, learningRate: learningRate_)
                try trainLinearModel(
                    trainer: trainer, trainFrame: trainFrame, model: &model, optimizer: &optimizer,
                    modelName: modelName, loss: linearModelLoss_, lossName: linearModelLossName, margin: margin_
                )
            } else {
                throw OptimizerError.unsupportedOptimizer(message: "Optimizer \(optimizerName) is not supported for model \(modelName)")
            }
        }
        
        // Initialize command-line argument mappings

        let embeddingsTensorDimensionalityReducers: [EmbeddingsDimensionalityReducer: (Tensor<Float>) -> Tensor<Float>]  = [
            .sum: { embeddings in
                embeddings.sum(alongAxes: [1])
            },
            .avg : { embeddings in
                embeddings.mean(alongAxes: [1])
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

        let defaultTasks: [Model: Task]  = [
            .transe: .linkPrediction,
            .transd : .linkPrediction,
            .rotate: .linkPrediction,
            .gcn: .classification,
            .vgae: .classification,
            .conve: .classification,
            .qrescal: .linkPrediction
        ]

        let defaultLosses: [Model: LinearModelLoss]  = [
            .transe: .sum,
            .transd : .sum,
            .rotate: .sigmoid
        ]

        let defaultOptimizers: [Model: Optimizer]  = [
            .transe: .adam,
            .transd : .adam,
            .rotate: .adam,
            .gcn: .adam,
            .vgae: .adam,
            .conve: .adam,
            .qrescal: .sgd
        ]

        // let lossFunctions: [LinearModelLoss: (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float>]  = [
        //     .sum: computeSumLoss,
        //     .sigmoid: computeSigmoidLoss
        // ]

        if task == .defaultTask {
            task = defaultTasks[model]!
        }

        if linearModelLoss == .defaultLoss {
            linearModelLoss = defaultLosses[model] ?? .sum
        }

        if optimizer == .defaultOptimizer {
            optimizer = defaultOptimizers[model]!
        }

        let linearModelLoss_ = linearModelLoss
        let margin_ = margin
        let optimizer_ = optimizer

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
        let languageModelName_ = languageModelName

        let graphRepresentation_ = graphRepresentation

        let entityIndicesGetters: [GraphRepresentation: (LabelFrame<Int32>) -> Tensor<Int32>]  = [
            .adjacencyMatrix: { labels in
                labels.indices
            },
            .adjacencyPairsMatrix : { labels in
                dataset_.getAdjacencyPairsIndices(labels: labels)
            }
        ]

        let stackedEmbeddingsWidth_ = stackedEmbeddingsWidth
        let stackedEmbeddingsHeight_ = stackedEmbeddingsHeight

        let convolutionFilterWidth_ = convolutionFilterWidth
        let convolutionFilterHeight_ = convolutionFilterHeight

        let nConvolutionFilters_ = nConvolutionFilters

        let hiddenLayerSize_ = hiddenLayerSize

        let modelName = model.rawValue
        let taskName = task.rawValue
        let linearModelLossName = linearModelLoss.rawValue
        let optimizerName = optimizer.rawValue

        // var result = [String: Float]()
        if (model == .qrescal) {
            print("Running quantum model...")
            if (task == .linkPrediction) {
                result += try GenericCVTester<QRescal<String, Int32>, TripleFrame<Int32>, QuantumTrainer<String, Int32>>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(
                    dataset: dataset, metrics: metrics, enableParallelism: false
                ) { trainer, trainFrame in
                    var model_ = QRescal(dimensionality: 64, dataset: dataset)
                    trainer.train(model: &model_, frame: trainFrame)
                    return model_
                } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                    (metric as! Metric).compute(model: model, trainFrame: trainLabels, testFrame: testLabels, dataset: dataset)    
                } getSamplesList: { dataset in
                    dataset.normalizedFrame
                }
            } else {
                throw TaskError.unsupportedTask(message: "Chosen task is not supported for the quantum model")
            }
        }
        else if (model == .conve) {
            if openke {
                let modelName = model.rawValue
                throw ModelError.unsupportedModel(message: "Model \(modelName) is not implemented in the OpenKE library!")
            }
            if task == .classification {
                let dataset_ = KnowledgeGraphDataset<String, Int32>(path: datasetPath, classes: labelsPath, texts: textsPath, device: device)
                let modelSavingLock = NSLock()
                var embeddingsWereInitialized: Bool = false
                
                result += try GenericCVTester<DenseClassifier<ConvE<String, Int32>, String, Int32>, LabelFrame<Int32>, ClassificationTrainer>(
                    nFolds: nFolds, nEpochs: classifierNEpochs, batchSize: batchSize
                ).test(
                    dataset: dataset_,
                    metrics: classificationMetrics,
                    enableParallelism: false
                ) { trainer, labels -> DenseClassifier<ConvE<String, Int32>, String, Int32> in
                
                    // 1. Train the base model with node encodings if necessary
                    
                    let shouldInitializeEmbeddings = !embeddingsWereInitialized && !readEmbeddings_
                    var model_ = !shouldInitializeEmbeddings ? try ConvE(dataset: dataset_, device: device) : ConvE(
                        embeddingDimensionality: embeddingDimensionality_,
                        stackedEmbeddingsWidth: stackedEmbeddingsWidth_,
                        stackedEmbeddingsHeight: stackedEmbeddingsHeight_,
                        filterWidth: convolutionFilterWidth_,
                        filterHeight: convolutionFilterHeight_,
                        nConvolutionalFilters: nConvolutionFilters_,
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
                        modelSavingLock.lock()
                        try model_.save()
                        modelSavingLock.unlock()
                        embeddingsWereInitialized = true
                    }

                    // 2. Train classification block

                    return try trainDenseClassifier(
                        graphEmbedder: model_,
                        dataset: dataset_,
                        classifierLearningRate: classifierLearningRate_,
                        labels: labels,
                        trainer: trainer,
                        getEntityIndices: entityIndicesGetters[graphRepresentation_]!,
                        device: device,
                        languageModelName: languageModelName_
                    )
                } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                    let logits = getLogits(model: model, getEntityIndices: entityIndicesGetters[graphRepresentation_]!, labels: testLabels)
                    return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
                } getSamplesList: { dataset in
                    dataset.labelFrame!
                }
            } else {
                throw TaskError.unsupportedTask(message: "Task \(taskName) is not implemented for model \(modelName)")
            }
        } else if (model == .vgae) {
            if openke {
                let modelName = model.rawValue
                throw ModelError.unsupportedModel(message: "Model \(modelName) is not implemented in the OpenKE library!")
            }
            if task == .classification {
                let embeddingsDimensionalityReducer_ = embeddingsDimensionalityReducer
                let graphRepresentation_ = graphRepresentation
                let modelSavingLock = NSLock()

                var embeddingsWereInitialized: Bool = false
                result += try GenericCVTester<DenseClassifier<VGAE<String, Int32>, String, Int32>, LabelFrame<Int32>, ClassificationTrainer>(nFolds: nFolds, nEpochs: classifierNEpochs, batchSize: batchSize).test(
                    dataset: dataset_,
                    metrics: classificationMetrics,
                    enableParallelism: false
                ) { trainer, labels -> DenseClassifier<VGAE<String, Int32>, String, Int32> in

                    // 1. Train the base model with node encodings if necessary
                    
                    let shouldInitializeEmbeddings = !embeddingsWereInitialized && !readEmbeddings_
                    var model_ = !shouldInitializeEmbeddings ? try VGAE(dataset: dataset, device: device) : VGAE(
                        embeddingDimensionality: embeddingDimensionality_,
                        dataset: dataset_,
                        device: device,
                        adjacencyTensorPath: adjacencyTensorPaths[graphRepresentation_]!
                    )
                    if shouldInitializeEmbeddings {
                        let trainer_ = ConvolutionAdjacencyTrainer(nEpochs: nEpochs_)
                        let optimizer_ = Adam<VGAE<String, Int32>>(for: model_, learningRate: learningRate_)
                        trainer_.train(dataset: dataset, model: &model_, optimizer: optimizer_,
                            trainTensorPath: trainTensorPaths[graphRepresentation_]!, adjacencyTensorPath: adjacencyTensorPaths[graphRepresentation_]!
                        )
                        modelSavingLock.lock()
                        try model_.save()
                        modelSavingLock.unlock()
                        embeddingsWereInitialized = true
                    }

                    // 2. Train classification block

                    return try trainDenseClassifier(
                        graphEmbedder: model_,
                        dataset: dataset_,
                        classifierLearningRate: classifierLearningRate_,
                        labels: labels,
                        trainer: trainer,
                        getEntityIndices: entityIndicesGetters[graphRepresentation_]!,
                        device: device,
                        languageModelName: languageModelName_,
                        reduceEmbeddingsTensorDimensionality: embeddingsTensorDimensionalityReducers[embeddingsDimensionalityReducer_]!,
                        shouldExpandTextEmbeddings: graphRepresentation_ != .adjacencyMatrix
                    )
                } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                    let logits = getLogits(model: model, getEntityIndices: entityIndicesGetters[graphRepresentation_]!, labels: testLabels)
                    return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
                } getSamplesList: { dataset in
                    dataset.labelFrame!
                }
            } else {
                throw TaskError.unsupportedTask(message: "Task \(taskName) is not implemented for model \(modelName)")
            }
        } else if (model == .gcn) {
            if openke {
                let modelName = model.rawValue
                throw ModelError.unsupportedModel(message: "Model \(modelName) is not implemented in the OpenKE library!")
            }
            if task == .classification {
                let dataset_ = KnowledgeGraphDataset<String, Int32>(path: datasetPath, classes: labelsPath, device: device)
                let graphAdjacencyMatrixGetter = trainTensorPaths[graphRepresentation_]!
                result += try GenericCVTester<GCN<String, Int32>, LabelFrame<Int32>, ConvolutionClassificationTrainer>(
                        nFolds: nFolds,
                        nEpochs: nEpochs,
                        batchSize: batchSize
                ).test(dataset: dataset_, metrics: classificationMetrics, enableParallelism: false) { trainer, labels in
                    var model_ = GCN(
                        embeddingDimensionality: embeddingDimensionality_,
                        dataset: dataset_,
                        device: device,
                        hiddenLayerSize: hiddenLayerSize_,
                        adjacencyTensorPath: adjacencyTensorPaths[graphRepresentation_]!
                    )
                    var optimizer = Adam<GCN<String, Int32>>(for: model_, learningRate: learningRate_)
                    trainer.train(dataset: dataset_, model: &model_, optimizer: &optimizer, labels: labels, getAdjacencyMatrix: graphAdjacencyMatrixGetter)
                    return model_
                } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                    let logits = model(dataset[keyPath: graphAdjacencyMatrixGetter]).flattened().gathering(atIndices: testLabels.indices)
                    return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
                } getSamplesList: { dataset in
                    dataset.labelFrame!
                }
            } else {
                throw TaskError.unsupportedTask(message: "Task \(taskName) is not implemented for model \(modelName)")
            }
        } else if (model == .rotate) {
            if openke {
                throw ModelError.unsupportedModel(message: "Model \(modelName) is not implemented in the OpenKE library!")
            } else {
                if task == .linkPrediction {
                    result += try GenericCVTester<RotatE<String, Int32>, TripleFrame<Int32>, LinearTrainer>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(
                        dataset: dataset, metrics: metrics, enableParallelism: false
                    ) { trainer, trainFrame in
                        var model_ = RotatE(embeddingDimensionality: embeddingDimensionality_, dataset: dataset, device: device)
                        try trainLinearModelUsingOptimizer(model: &model_, trainer: trainer, trainFrame: trainFrame)
                        return model_
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        (metric as! Metric).compute(model: model, trainFrame: trainLabels, testFrame: testLabels, dataset: dataset)    
                    } getSamplesList: { dataset in
                        dataset.normalizedFrame
                    }
                } else if task == .classification {
                    let modelSavingLock = NSLock()
                    var embeddingsWereInitialized: Bool = false

                    result += try GenericCVTester<DenseClassifier<RotatE<String, Int32>, String, Int32>, LabelFrame<Int32>, ClassificationTrainer>(
                        nFolds: nFolds, nEpochs: classifierNEpochs, batchSize: batchSize
                    ).test(
                        dataset: dataset_,
                        metrics: classificationMetrics,
                        enableParallelism: false
                    ) { trainer, labels -> DenseClassifier<RotatE<String, Int32>, String, Int32> in
                    
                        // 1. Train the base model with node encodings if necessary
                        
                        let shouldInitializeEmbeddings = !embeddingsWereInitialized && !readEmbeddings_
                        var model_ = !shouldInitializeEmbeddings ? try RotatE(dataset: dataset_, device: device) : RotatE(
                            embeddingDimensionality: embeddingDimensionality_,
                            dataset: dataset,
                            device: device
                        )
                        if shouldInitializeEmbeddings {
                            let trainer_ = LinearTrainer(nEpochs: nEpochs_, batchSize: batchSize_)
                            try trainLinearModelUsingOptimizer(model: &model_, trainer: trainer_, trainFrame: dataset.normalizedFrame)
                            modelSavingLock.lock()
                            try model_.save()
                            modelSavingLock.unlock()
                            embeddingsWereInitialized = true
                        }

                        // 2. Train classification block
                        
                        return try trainDenseClassifier(
                            graphEmbedder: model_,
                            dataset: dataset_,
                            classifierLearningRate: classifierLearningRate_,
                            labels: labels,
                            trainer: trainer,
                            getEntityIndices: entityIndicesGetters[graphRepresentation_]!,
                            device: device,
                            languageModelName: languageModelName_
                        )
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        let logits = getLogits(model: model, getEntityIndices: entityIndicesGetters[graphRepresentation_]!, labels: testLabels)
                        return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
                    } getSamplesList: { dataset in
                        dataset.labelFrame!
                    }
                } else {
                    throw TaskError.unsupportedTask(message: "Task \(taskName) is not implemented for model \(modelName)")
                }
            }
        } else if (model == .transe) {
            if openke {
                if task == .linkPrediction{
                    result += try GenericCVTester<OpenKEModel, TripleFrame<Int32>, OpenKEModelTrainer>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(
                        dataset: dataset, metrics: metrics, enableParallelism: false
                    ) { trainer, trainFrame in
                        OpenKEModel(
                            configuration: trainer.train(model: modelName, frame: trainFrame, dataset: dataset)
                        )
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        (metric as! Metric).compute(model: model, trainFrame: trainLabels, testFrame: testLabels, dataset: dataset)    
                    } getSamplesList: { dataset in
                        dataset.normalizedFrame
                    }
                } else {
                    throw TaskError.unsupportedTask(message: "OpenKE models do not support task \(taskName)")
                }
            } else {
                if task == .linkPrediction {
                    result += try GenericCVTester<TransE<String, Int32>, TripleFrame<Int32>, LinearTrainer>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(
                        dataset: dataset, metrics: metrics, enableParallelism: false
                    ) { trainer, trainFrame in
                        var model_ = TransE(embeddingDimensionality: embeddingDimensionality_, dataset: dataset, device: device)
                        try trainLinearModelUsingOptimizer(model: &model_, trainer: trainer, trainFrame: trainFrame)
                        return model_
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        (metric as! Metric).compute(model: model, trainFrame: trainLabels, testFrame: testLabels, dataset: dataset)    
                    } getSamplesList: { dataset in
                        dataset.normalizedFrame
                    }
                } else if task == .classification {
                    let modelSavingLock = NSLock()
                    var embeddingsWereInitialized: Bool = false

                    result += try GenericCVTester<DenseClassifier<TransE<String, Int32>, String, Int32>, LabelFrame<Int32>, ClassificationTrainer>(
                        nFolds: nFolds, nEpochs: classifierNEpochs, batchSize: batchSize
                    ).test(
                        dataset: dataset_,
                        metrics: classificationMetrics,
                        enableParallelism: false
                    ) { trainer, labels -> DenseClassifier<TransE<String, Int32>, String, Int32> in
                    
                        // 1. Train the base model with node encodings if necessary
                        
                        let shouldInitializeEmbeddings = !embeddingsWereInitialized && !readEmbeddings_
                        var model_ = !shouldInitializeEmbeddings ? try TransE(dataset: dataset_, device: device) : TransE(
                            embeddingDimensionality: embeddingDimensionality_,
                            dataset: dataset,
                            device: device
                        )
                        if shouldInitializeEmbeddings {
                            let trainer_ = LinearTrainer(nEpochs: nEpochs_, batchSize: batchSize_)
                            try trainLinearModelUsingOptimizer(model: &model_, trainer: trainer_, trainFrame: dataset.normalizedFrame)
                            modelSavingLock.lock()
                            try model_.save()
                            modelSavingLock.unlock()
                            embeddingsWereInitialized = true
                        }

                        // 2. Train classification block

                        return try trainDenseClassifier(
                            graphEmbedder: model_,
                            dataset: dataset_,
                            classifierLearningRate: classifierLearningRate_,
                            labels: labels,
                            trainer: trainer,
                            getEntityIndices: entityIndicesGetters[graphRepresentation_]!,
                            device: device,
                            languageModelName: languageModelName_
                        )
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        let logits = getLogits(model: model, getEntityIndices: entityIndicesGetters[graphRepresentation_]!, labels: testLabels)
                        return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
                    } getSamplesList: { dataset in
                        dataset.labelFrame!
                    }
                } else {
                    throw TaskError.unsupportedTask(message: "Task \(taskName) is not implemented for model \(modelName)")
                }
            }
        } else if (model == .transd) {
            if openke {
                if task == .linkPrediction {
                    result += try GenericCVTester<OpenKEModel, TripleFrame<Int32>, OpenKEModelTrainer>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(
                        dataset: dataset, metrics: metrics, enableParallelism: false
                    ) { trainer, trainFrame in
                        OpenKEModel(
                            configuration: trainer.train(model: modelName, frame: trainFrame, dataset: dataset)
                        )
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        (metric as! Metric).compute(model: model, trainFrame: trainLabels, testFrame: testLabels, dataset: dataset)    
                    } getSamplesList: { dataset in
                        dataset.normalizedFrame
                    }
                } else {
                    throw TaskError.unsupportedTask(message: "OpenKE models do not support task \(taskName)")
                }
            } else {
                if task == .linkPrediction {
                    result += try GenericCVTester<TransD<String, Int32>, TripleFrame<Int32>, LinearTrainer>(nFolds: nFolds, nEpochs: nEpochs, batchSize: batchSize).test(
                        dataset: dataset, metrics: metrics, enableParallelism: false
                    ) { trainer, trainFrame in
                        var model_ = TransD(embeddingDimensionality: embeddingDimensionality_, dataset: dataset, device: device)
                        try trainLinearModelUsingOptimizer(model: &model_, trainer: trainer, trainFrame: trainFrame)
                        return model_
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        (metric as! Metric).compute(model: model, trainFrame: trainLabels, testFrame: testLabels, dataset: dataset)    
                    } getSamplesList: { dataset in
                        dataset.normalizedFrame
                    }
                } else if task == .classification {
                    let modelSavingLock = NSLock()
                    var embeddingsWereInitialized: Bool = false

                    result += try GenericCVTester<DenseClassifier<TransD<String, Int32>, String, Int32>, LabelFrame<Int32>, ClassificationTrainer>(
                        nFolds: nFolds, nEpochs: classifierNEpochs, batchSize: batchSize
                    ).test(
                        dataset: dataset_,
                        metrics: classificationMetrics,
                        enableParallelism: false
                    ) { trainer, labels -> DenseClassifier<TransD<String, Int32>, String, Int32> in
                    
                        // 1. Train the base model with node encodings if necessary
                        
                        let shouldInitializeEmbeddings = !embeddingsWereInitialized && !readEmbeddings_
                        var model_ = !shouldInitializeEmbeddings ? try TransD(dataset: dataset_, device: device) : TransD(
                            embeddingDimensionality: embeddingDimensionality_,
                            dataset: dataset,
                            device: device
                        )
                        if shouldInitializeEmbeddings {
                            let trainer_ = LinearTrainer(nEpochs: nEpochs_, batchSize: batchSize_)
                            try trainLinearModelUsingOptimizer(model: &model_, trainer: trainer_, trainFrame: dataset.normalizedFrame)
                            modelSavingLock.lock()
                            try model_.save()
                            modelSavingLock.unlock()
                            embeddingsWereInitialized = true
                        }

                        // 2. Train classification block

                        return try trainDenseClassifier(
                            graphEmbedder: model_,
                            dataset: dataset_,
                            classifierLearningRate: classifierLearningRate_,
                            labels: labels,
                            trainer: trainer,
                            getEntityIndices: entityIndicesGetters[graphRepresentation_]!,
                            device: device,
                            languageModelName: languageModelName_
                        )
                    } computeMetric: { model, metric, trainLabels, testLabels, dataset -> Float in
                        let logits = getLogits(model: model, getEntityIndices: entityIndicesGetters[graphRepresentation_]!, labels: testLabels)
                        return (metric as! ClassificationMetric).compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
                    } getSamplesList: { dataset in
                        dataset.labelFrame!
                    }
                } else {
                    throw TaskError.unsupportedTask(message: "Task \(taskName) is not implemented for model \(modelName)")
                }
            }
        } else {
            throw ModelError.unsupportedModel(message: "Model \(modelName) is not supported yet!")
        }
    }
}

public struct ReportEntry {
    public let header: String
    public var metrics: [String: Float]
}

struct RestructureReport: ParsableCommand, Encodable {

    enum CodingKeys: String, CodingKey {
        case inputPath
        case outputPath
        case nDecimalPlaces
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(inputPath, forKey: .inputPath)
        try container.encode(outputPath, forKey: .outputPath)
        try container.encode(nDecimalPlaces, forKey: .nDecimalPlaces)
    }

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

        while offset < lines.count {
            skipEmptyLines()
            let header = lines[offset]
            offset += 1
            if offset >= lines.count {
                break
            }
            skipEmptyLines()
            var metrics = [String: Float]()
            while lines[offset] != "" {
                let metricWithValue = lines[offset].components(separatedBy: ": ")
                metrics[metricWithValue[0]] = metricWithValue[1] != "nan" ? Float(metricWithValue[1])! : Float.nan
                offset += 1
            }
            reportEntries.append(ReportEntry(header: header, metrics: metrics))
        }
        return reportEntries
    }

    private func writeData(path: String, data: [ReportEntry]) throws {
        let metric_keys = data[0].metrics.keys.sorted()
        let model_keys = data.map {
            $0.header
        }
        let lines = ["metric\t\(model_keys.joined(separator: "\t"))"] + metric_keys.map { metric in
            "\(metric)\t\(model_keys.map{data[$0]!.metrics[metric]!.isNaN ? "-" : String(format: "%.\(nDecimalPlaces)f", data[$0]!.metrics[metric]!)}.joined(separator: "\t"))"
        }
        try writeLines(path: path, lines: lines)
    }

    mutating func run(_ result: inout [String: Any]) throws {
        let data = try readData(path: inputPath)
        try writeData(path: outputPath, data: data)
    }
}

struct Agk: ParsableCommand {
    static var configuration = CommandConfiguration(
            abstract: "A tool for automating operation on the knowledge graph models",
            subcommands: [CrossValidate.self, RestructureReport.self, StartServer.self],
            defaultSubcommand: CrossValidate.self
    )
}

Agk.main()
