import Foundation
import TensorFlow

public protocol GenericModel {
    associatedtype Scalar: TensorFlowScalar
    func callAsFunction(_ triples: Tensor<Scalar>) -> Tensor<Float>
}
