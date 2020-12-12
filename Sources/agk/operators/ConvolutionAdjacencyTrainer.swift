import Foundation
import TensorFlow
import Checkpoints

public struct ConvolutionAdjacencyTrainer {
    public let nEpochs: Int
    public let epsilon: Float

    public init(nEpochs: Int, epsilon: Float = 0.001) {
        self.nEpochs = nEpochs
        self.epsilon = epsilon
    }

    public func train<Model, SourceElement, NormalizedElement>(
            dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>,
            model: inout Model,
            optimizer: Adam<Model>,
            trainTensorPath: KeyPath<KnowledgeGraphDataset<SourceElement, NormalizedElement>,  Tensor<Float>> = \.tunedAdjecencyMatrixInverse,
            adjacencyTensorPath: KeyPath<KnowledgeGraphDataset<SourceElement, NormalizedElement>,  Tensor<Int8>> = \.frame.adjacencyTensor
    ) where Model: ConvolutionGraphModel, Model.Scalar == Float {
        for i in 1...nEpochs {
            var losses: [Float] = []
            // for batch in dataset.frame.batched(size: batchSize) {
            let adjacencyTensor = dataset[keyPath: adjacencyTensorPath]
            let expectedAdjacencyMatrix = Tensor<Float>(adjacencyTensor - adjacencyTensor.diagonalPart().diagonal())
            let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let generatedAdjacencyMatrix = model(dataset[keyPath: trainTensorPath])
                // print(generatedAdjacencyMatrix.unstacked())
                // print(expectedAdjacencyMatrix.unstacked())
                return sigmoidCrossEntropy(logits: generatedAdjacencyMatrix + epsilon, labels: expectedAdjacencyMatrix + epsilon)
                // (expectedAdjacencyMatrix * log(expectedAdjacencyMatrix / generatedAdjacencyMatrix + 1.0)).sum()
            }
            optimizer.update(&model, along: grad)
            model = model.normalizeEmbeddings()
            losses.append(loss.scalarized())
            // }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
