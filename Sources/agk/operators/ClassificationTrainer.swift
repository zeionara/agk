import Foundation
import TensorFlow
import Checkpoints
// import TextModels

public struct ClassificationTrainer: Trainer {
    public let nEpochs: Int
    public let batchSize: Int
    // public let textEmbedder: ELMO?

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
        // do {
        //     self.textEmbedder = try ELMO(getModelsCacheRoot())
        // } catch {
        //     self.textEmbedder = Optional.none
        // }
    }

    public func train<OptimizerType, GraphModelType>(
        model: inout DenseClassifier<GraphModelType, String, Int32>, optimizer: inout OptimizerType, labels: LabelFrame<Int32>, getEntityIndices: (LabelFrame<Int32>) -> Tensor<Int32> = { $0.indices }
    ) where OptimizerType: Optimizer, OptimizerType.Model == DenseClassifier<GraphModelType, String, Int32> {
        for i in 1...nEpochs{
            var losses: [Float] = []
            for batch in labels.batched(size: batchSize) {
                let entityIndices = getEntityIndices(batch)
                // let texts = entityIndices.unstacked().map{model.dataset.entityId2Text[$0.scalar!]}
                // let textEmbeddings = Tensor(stacking: model.textEmbedder.embed(texts).map{tensor in unpackOptionalTensor(tensor, model.textEmbedder.embeddingSize)})
                // if entityIndices.shape.count == 1 {
                    // let texts = entityIndices.unstacked().map{model.dataset.entityId2Text[$0.scalar!]}
                    // textEmbeddings = Tensor(stacking: textEmbedder!.embed(texts).map{tensor in unpackOptionalTensor(tensor, textEmbedder!.embeddingSize)})
                    // print(textEmbeddings.shape)
                    // print(dataset.entityId2Text)
                // }
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    // print("Computing model labels")
                    let labels_ = model(entityIndices) // model(Tensor<Int32>(batch.adjacencyTensor))
                    // print("Computing loss")
                    return sigmoidCrossEntropy(
                        logits: labels_.flattened(),
                        labels: batch.labels
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
