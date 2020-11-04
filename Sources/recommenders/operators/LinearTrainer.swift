import Foundation
import TensorFlow


public struct LinearTrainer: Trainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model, OptimizerType>(
            frame: TripleFrame<Int32>, model: inout Model, optimizer: inout OptimizerType, margin: Float = 2.0,
            loss: @differentiable (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float> = computeSumLoss
    ) where Model: LinearGraphModel, OptimizerType: Optimizer, OptimizerType.Model == Model {
        for i in 1...nEpochs {
            var losses: [Float] = []
            for batch in frame.batched(size: batchSize) {
                let negativeFrame = batch.sampleNegativeFrame(negativeFrame: frame.negative)
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    loss(model(batch.tensor), model(negativeFrame.tensor), margin)
                }
                optimizer.update(&model, along: grad)
                model = model.normalizeEmbeddings() as! OptimizerType.Model
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
