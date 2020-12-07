import Foundation
import TensorFlow

public protocol GenericModel {
    func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float>
}
