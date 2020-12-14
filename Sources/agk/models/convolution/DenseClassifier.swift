import Foundation
import TensorFlow
import Checkpoints

public struct DenseClassifier<GraphEmbedderType>: GenericModel, Module where GraphEmbedderType: ConvolutionGraphModel {
    @noDerivative public var graphEmbedder: GraphEmbedderType
    private var layer: Dense<Float>
    private var inputLayer: Dense<Float>
    private var dropout: GaussianDropout<Float>
    @noDerivative private let reduceEmbeddingsTensorDimensionality: (Tensor<Float>) -> Tensor<Float>
    @noDerivative public let device: Device

    public init(
        graphEmbedder: GraphEmbedderType,
        device: Device = Device.default,
        activation: @escaping Dense<Float>.Activation = relu,
        reduceEmbeddingsTensorDimensionality: @escaping (Tensor<Float>) -> Tensor<Float> = { $0.sum(alongAxes: [1]) }
    ) {
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
    }

    @differentiable
    public func callAsFunction(
        _ entityIds: Tensor<Int32>
    ) -> Tensor<Float> {
        return sigmoid(
            layer(
                dropout(
                    inputLayer(
                        entityIds.shape.count == 1 ? graphEmbedder.entityEmbeddings(entityIds) : reduceEmbeddingsTensorDimensionality(graphEmbedder.entityEmbeddings(entityIds)).reshaped(to: [entityIds.shape[0], -1])
                    )
                )
            )
        )
    }
}
