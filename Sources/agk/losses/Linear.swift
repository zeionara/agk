import Foundation
import TensorFlow

public typealias LinearModelLossImpl = (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float>

@differentiable
public func computeSumLoss(_ positiveScores: Tensor<Float>, _ negativeScores: Tensor<Float>, _ margin: Float = 2.0) -> Tensor<Float> {
    max(0, margin + positiveScores.sum() - negativeScores.sum())
}

@differentiable
public func inverseSigmoidLog(tensor: Tensor<Float>) -> Tensor<Float> {
    -log(sigmoid(tensor))
}

@differentiable
public func computeSigmoidLoss(_ positiveScores: Tensor<Float>, _ negativeScores: Tensor<Float>, _ margin: Float = 2.0) -> Tensor<Float> {
    inverseSigmoidLog(tensor: margin - positiveScores).sum() + inverseSigmoidLog(tensor: negativeScores - margin).sum()
}
