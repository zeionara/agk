import Foundation
import TensorFlow


public struct ConvolutionClassificationTrainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model>(dataset: KnowledgeGraphDataset, model: inout Model, optimizer: Adam<Model>) where Model: ConvolutionGraphModel {
        for i in 1...nEpochs{
            var losses: [Float] = []
            for batch in dataset.frame.batched(size: batchSize) {
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    let labels_ = model(Tensor<Int32>(batch.adjacencyTensor))
                    return sigmoidCrossEntropy(logits: labels_.flattened().gathering(atIndices: dataset.labelFrame!.indices), labels: dataset.labelFrame!.labels)
                }
                optimizer.update(&model, along: grad)
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}