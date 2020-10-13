import Foundation
import TensorFlow


public struct ConvolutionAdjacencySequenceTrainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model>(dataset: KnowledgeGraphDataset, model: inout Model, optimizer: Adam<Model>) where Model: ConvolutionGraphModel {
        for i in 1...nEpochs {
            var losses: [Float] = []
            for batch in dataset.frame.batched(size: batchSize) {
                let expectedAdjacencySequence = dataset.normalizedFrame.adjacencySequence
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    let generatedAdjacencySequence = model(Tensor<Int32>(dataset.normalizedFrame.tensor))
                    return sigmoidCrossEntropy(logits: generatedAdjacencySequence, labels: expectedAdjacencySequence)
                }
                optimizer.update(&model, along: grad)
//                model = model.normalizeEmbeddings()
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
