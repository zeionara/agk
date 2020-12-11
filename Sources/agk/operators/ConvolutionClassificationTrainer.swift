import Foundation
import TensorFlow
import Checkpoints

public struct ConvolutionClassificationTrainer {
    public let nEpochs: Int

    public init(nEpochs: Int) {
        self.nEpochs = nEpochs
    }

    public func train<Model, OptimizerType, SourceElement>(
        dataset: KnowledgeGraphDataset<SourceElement, Int32>, model: inout Model, optimizer: inout OptimizerType,
        labels: LabelFrame<Int32>, frame: TripleFrame<Int32>
    ) where Model: ConvolutionGraphModel, Model.Scalar == Float, OptimizerType: Optimizer, OptimizerType.Model == Model {
        for i in 1...nEpochs{
            var losses: [Float] = []
            let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                print("Computing model labels")
                let labels_ = model(dataset.tunedAdjecencyMatrixInverse) // model(Tensor<Int32>(batch.adjacencyTensor))
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
