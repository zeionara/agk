import Foundation
import TensorFlow
import Checkpoints
import TextModels

public struct DenseClassifier<GraphEmbedderType, SourceElement, NormalizedElement>: GenericModel, Module where GraphEmbedderType: ConvolutionGraphModel, SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
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
        unpackOptionalTensor: (Tensor<Float>?, Int) -> Tensor<Float> = {$0 == Optional.none ? Tensor<Float>(zeros: [$1]) : $0!}
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
            textEmbeddings = Embedding<Float>(
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
        return sigmoid(
            layer(
                dropout(
                    inputLayer(
                        entityEmbeddings
                    )
                )
            )
        )
    }
}
