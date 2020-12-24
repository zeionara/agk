import Foundation
import TensorFlow
import Logging

public struct LinearTrainer: Trainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model, OptimizerType>(
            frame: TripleFrame<Int32>, model: inout Model, optimizer: inout OptimizerType, logger: Logger, margin: Float = 2.0,
            loss: @differentiable (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float> = computeSumLoss
    ) where Model: LinearGraphModel, OptimizerType: Optimizer, OptimizerType.Model == Model, Model.Scalar == Int32 {
        for i in 1...nEpochs {
            var losses: [Float] = []
            for batch in frame.batched(size: batchSize) {
                let negativeFrame = batch.sampleNegativeFrame(negativeFrame: frame.negative)
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    loss(model(batch.tensor), model(negativeFrame.tensor), margin)
                }
                optimizer.update(&model, along: grad)
                model = model.normalizeEmbeddings()
                losses.append(loss.scalarized())
            }
            logger.trace("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
