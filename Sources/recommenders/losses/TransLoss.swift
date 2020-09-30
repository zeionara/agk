import Foundation
import TensorFlow


public protocol TransLoss {
//    @differentiable(wrt: scores)
    func compute<Scalar>(scores: Tensor<Scalar>) -> Tensor<Scalar> where Scalar: TensorFlowFloatingPoint
}
