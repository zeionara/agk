import Foundation
import TensorFlow
import Checkpoints
import TextModels

// @differentiable
// public func average(lhs: Tensor<Float>, rhs: Tensor<Float>) -> Tensor<Float> {
//     return lhs
// }

public struct DenseClassifier<GraphEmbedderType, SourceElement, NormalizedElement>: GenericModel, Module where GraphEmbedderType: ConvolutionGraphModel, SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    @noDerivative public var graphEmbedder: GraphEmbedderType
    @noDerivative public var textEmbedder: ELMO
    private var layer: Dense<Float>
    private var inputLayer: Dense<Float>
    private var dropout: GaussianDropout<Float>
    private var textEmbeddings: Embedding<Float>
    private var textEmbeddingsReshaper: Dense<Float>
    @noDerivative public let dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>
    @noDerivative private let reduceEmbeddingsTensorDimensionality: (Tensor<Float>) -> Tensor<Float>
    // @noDerivative private let aggregateGraphAndTextEmbeddings: (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
    @noDerivative public let device: Device

    public init(
        graphEmbedder: GraphEmbedderType,
        dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>,
        device: Device = Device.default,
        activation: @escaping Dense<Float>.Activation = relu,
        reduceEmbeddingsTensorDimensionality: @escaping (Tensor<Float>) -> Tensor<Float> = { $0.sum(alongAxes: [1]) },
        unpackOptionalTensor: (Tensor<Float>?, Int) -> Tensor<Float> = {$0 == Optional.none ? Tensor<Float>(zeros: [$1]) : $0!}// ,
        // aggregateGraphAndTextEmbeddings: @differentiable @escaping (Tensor<Float>, Tensor<Float>) -> Tensor<Float> = average
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
        // self.aggregateGraphAndTextEmbeddings = aggregateGraphAndTextEmbeddings
        self.dataset = dataset
        let textEmbedder_ = try ELMO(getModelsCacheRoot())
        self.textEmbeddings = Embedding<Float>(
            embeddings: Tensor(
                stacking: textEmbedder_.embed(
                    dataset.normalizedFrame.entities.sorted().map{dataset.entityId2Text[$0]} // dataset.frame.entities.sorted()
                ).map{tensor in
                    unpackOptionalTensor(tensor, textEmbedder_.embeddingSize)
                }
            )
        )
        textEmbedder = textEmbedder_
        self.textEmbeddingsReshaper = Dense<Float>(
            copying: Dense<Float>(
                inputSize: textEmbedder_.embeddingSize,
                outputSize: graphEmbedder.entityEmbeddings.embeddings.shape[1],
                activation: activation
            ),
            to: device
        )
    }

    // @differentiable
    // private func getFlatEntityEmbeddings(embeddingsSource: Embedding<Float>, entityIds: Tensor<Int32>) -> Tensor<Float> {
    //     return entityIds.shape.count == 1 ?
    //         embeddingsSource(entityIds) :
    //         reduceEmbeddingsTensorDimensionality(embeddingsSource(entityIds)).reshaped(to: [entityIds.shape[0], -1])
    // }

    @differentiable
    public func callAsFunction(
        _ entityIds: Tensor<Int32> //,
        // _ textEmbeddings: Tensor<Float>
    ) -> Tensor<Float> {
        // if entityIds.shape.count == 1 {
        //     // print(entityIds.unstacked().map{dataset.entityId2Text[$0.scalar!]})
        //     print(dataset.entityId2Text)
        // }
        return sigmoid(
            layer(
                dropout(
                    inputLayer(
                        (
                            (
                                entityIds.shape.count == 1 ?
                                graphEmbedder.entityEmbeddings(entityIds) :
                                reduceEmbeddingsTensorDimensionality(graphEmbedder.entityEmbeddings(entityIds)).reshaped(to: [entityIds.shape[0], -1])
                            ) + (
                                entityIds.shape.count == 1 ?
                                textEmbeddings(entityIds) :
                                reduceEmbeddingsTensorDimensionality(graphEmbedder.entityEmbeddings(entityIds)).reshaped(to: [entityIds.shape[0], -1]) // FIXME
                            )
                        ) / 2
                    )
                )
            )
        )
    }
}
