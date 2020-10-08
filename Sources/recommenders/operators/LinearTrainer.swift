import Foundation
import TensorFlow


public struct LinearTrainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model>(dataset: KnowledgeGraphDataset, model: inout Model, optimizer: Adam<Model>, margin: Float = 2.0,
                             loss: @differentiable (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float> = computeSumLoss) where Model: LinearGraphModel {
        for i in 1...nEpochs {
            var losses: [Float] = []
            for batch in dataset.normalizedFrame.batched(size: batchSize) {
                let negativeFrame = batch.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame)
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    return loss(model(batch.tensor), model(negativeFrame.tensor), margin)
                }
                optimizer.update(&model, along: grad)
                model = model.normalizeEmbeddings() as! Model
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
