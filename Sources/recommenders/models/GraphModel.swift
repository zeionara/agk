import Foundation
import TensorFlow

public protocol GraphModel: Module {
    @differentiable
    func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float>
}
