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
            // for batch in frame.batched(size: batchSize) {
            // print("Generated batch")
            let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                print("Computing model labels")
                let labels_ = model(dataset.tunedAdjecencyMatrixInverse) // model(Tensor<Int32>(batch.adjacencyTensor))
                print("Computing loss")
                // print(labels_.flattened().gathering(atIndices: dataset.labelFrame!.indices))
                // print(dataset.labelFrame!.labels)
                return sigmoidCrossEntropy(
                    logits: labels_.flattened().gathering(atIndices: labels.indices) + 0.001,
                    labels: labels.labels + 0.001
                )
                // kullbackLeiblerDivergence(predicted: labels_.flattened().gathering(atIndices: dataset.labelFrame!.indices) + 0.001, expected: dataset.labelFrame!.labels + 0.001)
            }
            // print(grad)
            // print(grad)
            optimizer.update(&model, along: grad)
            losses.append(loss.scalarized())
            // }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
