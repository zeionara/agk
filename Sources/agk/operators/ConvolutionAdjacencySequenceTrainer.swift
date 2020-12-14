import Foundation
import TensorFlow


public struct ConvolutionAdjacencySequenceTrainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model, SourceElement>(dataset: KnowledgeGraphDataset<SourceElement, Int32>, model: inout Model, optimizer: Adam<Model>) where Model: ConvolutionGraphModel, Model.Scalar == Int32 {
        for i in 1...nEpochs {
            var losses: [Float] = []
            for batch in dataset.normalizedFrame.batched(size: batchSize) {
                let expectedAdjacencySequence = batch.adjacencySequence
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    let generatedAdjacencySequence = model(Tensor<Int32>(batch.tensor))
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
