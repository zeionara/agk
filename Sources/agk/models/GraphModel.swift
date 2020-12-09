import Foundation
import TensorFlow

public protocol GraphModel: Module, GenericModel {
    @differentiable
    func callAsFunction(_ triples: Tensor<Scalar>) -> Tensor<Float>
    func normalizeEmbeddings() -> Self
}
