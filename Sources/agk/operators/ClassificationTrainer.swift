import Foundation
import TensorFlow
import Checkpoints

public struct ClassificationTrainer: Trainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<OptimizerType, GraphModelType>(
        model: inout DenseClassifier<GraphModelType>, optimizer: inout OptimizerType, labels: LabelFrame<Int32>, getEntityIndices: (LabelFrame<Int32>) -> Tensor<Int32> = { $0.indices }
    ) where OptimizerType: Optimizer, OptimizerType.Model == DenseClassifier<GraphModelType> {
        for i in 1...nEpochs{
            var losses: [Float] = []
            for batch in labels.batched(size: batchSize) {
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    // print("Computing model labels")
                    let entityIndices = getEntityIndices(batch)
                    let labels_ = model(entityIndices) // model(Tensor<Int32>(batch.adjacencyTensor))
                    // print("Computing loss")
                    return sigmoidCrossEntropy(
                        logits: labels_.flattened() + 0.001,
                        labels: batch.labels + 0.001
                    )
                    //sigmoidCrossEntropy(
                    //    logits: labels_.flattened().gathering(atIndices: batch.indices) + 0.001,
                    //    labels: batch.labels + 0.001
                    //)
                }
                optimizer.update(&model, along: grad)
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}
