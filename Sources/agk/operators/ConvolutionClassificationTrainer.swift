import Foundation
import TensorFlow
import Checkpoints

public struct ConvolutionClassificationTrainer: Trainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int = -1) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model, OptimizerType, SourceElement>(
        dataset: KnowledgeGraphDataset<SourceElement, Int32>, model: inout Model, optimizer: inout OptimizerType,
        labels: LabelFrame<Int32>,
        getAdjacencyMatrix: KeyPath<KnowledgeGraphDataset<SourceElement, Int32>, Tensor<Float>> = \.tunedAdjecencyMatrixInverse
    ) where Model: ConvolutionGraphModel, Model.Scalar == Float, OptimizerType: Optimizer, OptimizerType.Model == Model {
        for i in 1...nEpochs{
            var losses: [Float] = []
            let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                print("Computing model labels")
                let labels_ = model(dataset[keyPath: getAdjacencyMatrix]) // model(Tensor<Int32>(batch.adjacencyTensor))
                print("Computing loss")
                return sigmoidCrossEntropy(
                    logits: labels_.flattened().gathering(atIndices: labels.indices) + 0.001,
                    labels: labels.labels + 0.001
                )
            }
            optimizer.update(&model, along: grad)
            losses.append(loss.scalarized())
            model = model.normalizeEmbeddings()
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
