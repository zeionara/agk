import Foundation
import TensorFlow


public struct ConvolutionAdjacencyTrainer {
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
                let expectedAdjacencyMatrix = Tensor<Float>(batch.adjacencyTensor - batch.adjacencyTensor.diagonalPart().diagonal())
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    let generatedAdjacencyMatrix = model(Tensor<Int32>(batch.adjacencyTensor))
                    return (expectedAdjacencyMatrix * log(expectedAdjacencyMatrix / generatedAdjacencyMatrix + 1.0)).sum() // kullbackLeiblerDivergence(predicted: generatedAdjacencyMatrix, expected: expectedAdjacencyMatrix)
                }
                optimizer.update(&model, along: grad)
                model = model.normalizeEmbeddings()
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
