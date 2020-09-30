import Foundation
import TensorFlow


public struct Trainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model>(dataset: KnowledgeGraphDataset, model: inout Model, optimizer: Adam<Model>, margin: Float = 2.0) where Model: GraphModel {
        for i in 1...nEpochs{
            var losses: [Float] = []
            for batch in dataset.normalizedFrame.batched(size: batchSize) {
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    max(0, margin + model(batch.tensor).sum() - model(batch.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame).tensor).sum())
                }
                optimizer.update(&model, along: grad)
                model = model.normalizeEmbeddings() as! Model
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}