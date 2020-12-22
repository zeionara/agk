import Foundation
import TensorFlow
import Checkpoints
import TextModels

public func prepareDenseClassifierOutputForMetricsComputation(_ output: Tensor<Float>) -> Tensor<Float> {
    return output.reshaped(to: [-1, 2]).transposed().unstacked()[0].replaceNans()
}

public func trainDenseClassifier<GraphEmbedderType>(
        graphEmbedder: GraphEmbedderType, dataset: KnowledgeGraphDataset<String, Int32>, classifierLearningRate: Float, labels: LabelFrame<Int32>,
        trainer: ClassificationTrainer,
        getEntityIndices: (LabelFrame<Int32>) -> Tensor<Int32> = { $0.indices }, device: Device = .default, languageModelName: String? = Optional.none,
        reduceEmbeddingsTensorDimensionality: @escaping (Tensor<Float>) -> Tensor<Float> = { $0.sum(alongAxes: [1]) },
        shouldExpandTextEmbeddings: Bool = false
) throws -> DenseClassifier<GraphEmbedderType, String, Int32> {
    while true {
        var classifier = try DenseClassifier(
            graphEmbedder: graphEmbedder,
            dataset: dataset,
            device: device,
            reduceEmbeddingsTensorDimensionality: reduceEmbeddingsTensorDimensionality,
            textEmbeddingModelName: languageModelName,
            shouldExpandTextEmbeddings: shouldExpandTextEmbeddings
        )
        var classificationOptimizer = Adam<DenseClassifier>(for: classifier, learningRate: classifierLearningRate)
        do {
            try trainer.train(model: &classifier, optimizer: &classificationOptimizer, labels: labels, getEntityIndices: getEntityIndices)
            return classifier
        } catch InitializationError.unsuccessfulInitialization {
            print("Unlucky initialization. Trying again...")
        }
    }
}

public func getLogits<GraphEmbedderType, SourceElement, NormalizedElement>(
        model: DenseClassifier<GraphEmbedderType, SourceElement, NormalizedElement>,
        getEntityIndices: (LabelFrame<Int32>) -> Tensor<Int32> = { $0.indices },
        labels: LabelFrame<Int32>
) -> Tensor<Float> {
    return prepareDenseClassifierOutputForMetricsComputation(
        model(
            getEntityIndices(
                labels
            )
        )
    )
}

public struct DenseClassifier<GraphEmbedderType, SourceElement, NormalizedElement>: GenericModel, Module where GraphEmbedderType: EntityEmbedder, SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    @noDerivative public var graphEmbedder: GraphEmbedderType
    @noDerivative public var textEmbedder: ELMO?
    private var layer: Dense<Float>
    private var inputLayer: Dense<Float>
    private var dropout: GaussianDropout<Float>
    @noDerivative private var textEmbeddings: Embedding<Float>?
    private var textEmbeddingsReshaper: Dense<Float>
    @noDerivative public let dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>
    @noDerivative private let reduceEmbeddingsTensorDimensionality: (Tensor<Float>) -> Tensor<Float>
    @noDerivative public let device: Device

    public init(
        graphEmbedder: GraphEmbedderType,
        dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>,
        device: Device = Device.default,
        activation: @escaping Dense<Float>.Activation = relu,
        reduceEmbeddingsTensorDimensionality: @escaping (Tensor<Float>) -> Tensor<Float> = { $0.sum(alongAxes: [1]) },
        textEmbeddingModelName: String? = Optional.none,
        unpackOptionalTensor: (Tensor<Float>?, Int) -> Tensor<Float> = {$0 == Optional.none ? Tensor<Float>(zeros: [$1]) : $0!},
        shouldExpandTextEmbeddings: Bool = false
    ) throws {
        self.graphEmbedder = graphEmbedder
        self.layer = Dense<Float>(
            copying: Dense<Float>(
                inputSize: graphEmbedder.entityEmbeddings.embeddings.shape[1],
                outputSize: 1,
                activation: activation
            ),
            to: device
        )
        self.inputLayer = Dense<Float>(
            copying: Dense<Float>(
                inputSize: graphEmbedder.entityEmbeddings.embeddings.shape[1],
                outputSize: graphEmbedder.entityEmbeddings.embeddings.shape[1],
                activation: activation
            ),
            to: device
        )
        self.dropout = GaussianDropout<Float>(probability: 0.1)
        self.device = device
        self.reduceEmbeddingsTensorDimensionality = reduceEmbeddingsTensorDimensionality
        self.dataset = dataset
        if let textEmbeddingModelName_ = textEmbeddingModelName {
            let textEmbedder_ = try ELMO(getModelsCacheRoot(), textEmbeddingModelName_)
            let textEmbeddings_ = Embedding<Float>(
                embeddings: Tensor(
                    stacking: textEmbedder_.embed(
                        dataset.normalizedFrame.entities.sorted().map{dataset.entityId2Text[$0]}
                    ).map{tensor in
                        unpackOptionalTensor(tensor, textEmbedder_.embeddingSize)
                    }
                )
            )
            textEmbedder = textEmbedder_
            textEmbeddingsReshaper = Dense<Float>(
                copying: Dense<Float>(
                    inputSize: textEmbedder_.embeddingSize,
                    outputSize: graphEmbedder.entityEmbeddings.embeddings.shape[1],
                    activation: activation
                ),
                to: device
            )
            if shouldExpandTextEmbeddings {
                let expandedEmbeddings = Tensor(
                    stacking: Array(
                        repeating: textEmbeddings_.embeddings,
                        count: graphEmbedder.entityEmbeddings.embeddings.shape[0] / textEmbeddings_.embeddings.shape[0] 
                    )
                ).transposed(permutation: [1, 0, 2]).reshaped(to: [-1, textEmbeddings_.embeddings.shape[1]])
                textEmbeddings = Embedding<Float>(
                    embeddings: expandedEmbeddings
                )
            } else {
                textEmbeddings = textEmbeddings_
            }
        } else {
            textEmbeddings = Optional.none
            textEmbedder = Optional.none
            textEmbeddingsReshaper = Dense<Float>(
                copying: Dense<Float>(
                    inputSize: 1,
                    outputSize: 1,
                    activation: activation
                ),
                to: device
            )
        }
    }

    @differentiable
    public func callAsFunction(
        _ entityIds: Tensor<Int32>
    ) -> Tensor<Float> {
        let entityEmbeddings: Tensor<Float>
        if let textEmbeddings_ = textEmbeddings {
            entityEmbeddings = (
                (
                    entityIds.shape.count == 1 ?
                    graphEmbedder.entityEmbeddings(entityIds) :
                    reduceEmbeddingsTensorDimensionality(graphEmbedder.entityEmbeddings(entityIds)).reshaped(to: [entityIds.shape[0], -1])
                ) + textEmbeddingsReshaper(
                    entityIds.shape.count == 1 ?
                    textEmbeddings_(entityIds) :
                    reduceEmbeddingsTensorDimensionality(textEmbeddings_(entityIds)).reshaped(to: [entityIds.shape[0], -1])
                )
            ) / 2
        } else {
            entityEmbeddings = entityIds.shape.count == 1 ?
                graphEmbedder.entityEmbeddings(entityIds) :
                reduceEmbeddingsTensorDimensionality(graphEmbedder.entityEmbeddings(entityIds)).reshaped(to: [entityIds.shape[0], -1])
        }
        let classifierOutput = layer(
            dropout(
                inputLayer(
                    entityEmbeddings
                )
            )
        )
        return softmax(
            Tensor(
                stacking: [
                    classifierOutput,
                    1 / classifierOutput
                ]
            ).transposed()
        )
    }
}
