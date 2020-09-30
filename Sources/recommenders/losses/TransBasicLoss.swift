import Foundation
import TensorFlow


public struct TransBasicLoss: TransLoss {
//    @differentiable(wrt: scores)
    public func compute<Scalar>(scores: Tensor<Scalar>) -> Tensor<Scalar> where Scalar: TensorFlowFloatingPoint {
        scores.sum()
    }
}
