import Foundation
import TensorFlow

public protocol GraphModel: Module {
//    associatedtype InputMatrixElement: TensorFlowScalar
    @differentiable
    func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float>
    func normalizeEmbeddings() -> Self
}
