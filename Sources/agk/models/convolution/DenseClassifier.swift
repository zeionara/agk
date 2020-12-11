import Foundation
import TensorFlow
import Checkpoints

public struct DenseClassifier<SourceElement, NormalizedElement>: Module where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    @noDerivative public var graphEmbedder: VGAE<SourceElement, NormalizedElement>
    private var layer: Dense<Float>
    @noDerivative public let device: Device

    public init(
        graphEmbedder: VGAE<SourceElement, NormalizedElement>,
        device: Device = Device.default,
        activation: @escaping Dense<Float>.Activation = relu
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
        self.device = device
    }

    @differentiable
    public func callAsFunction(_ entityIds: Tensor<Int32>) -> Tensor<Float> {
        return sigmoid(
            layer(
                graphEmbedder.entityEmbeddings(entityIds)
            )
        )
    }
}
