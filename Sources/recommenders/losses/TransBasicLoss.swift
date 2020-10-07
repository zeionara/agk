import Foundation
import TensorFlow


public struct TransBasicLoss: TransLoss {
//    @differentiable(wrt: scores)
    public func compute<Scalar>(scores: Tensor<Scalar>) -> Tensor<Scalar> where Scalar: TensorFlowFloatingPoint {
        scores.sum()
    }
}
//
//@differentiable(wrt: scores)
//public func compute<Scalar>(positiveScores: Tensor<Scalar>, negativeScores: Tensor<Scalar>) -> Tensor<Scalar> where Scalar: TensorFlowFloatingPoint {
//    max(0, margin + positiveScores.sum() - negativeScores.sum())
//}
