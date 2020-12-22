import Foundation
import TensorFlow
import Checkpoints

public extension Tensor where Scalar == Float {
    func replaceNans(with defaultValue: Tensor<Scalar> = Tensor(0.002)) -> Tensor<Scalar> {
        return Tensor(stacking:
            self.flattened().unstacked().map{ item -> Tensor<Scalar> in
                if item.scalar!.isNaN {
                    return defaultValue
                } else {
                    return item
                }
            }
        ).reshaped(to: self.shape)
    }
}

enum InitializationError: Error {
    case unsuccessfulInitialization(message: String)
}

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
    ) throws where OptimizerType: Optimizer, OptimizerType.Model == DenseClassifier<GraphModelType, String, Int32> {
        var isFirstEpoch = true
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
                    // print(labels_.reshaped(to: [-1, 2]))
                    // print(Tensor<Int32>(batch.labels))
                    let logits = labels_.reshaped(to: [-1, 2]) // .replaceNans()
                    let probabilities = Tensor(
                        stacking: [
                            batch.labels,
                            1 - batch.labels
                        ]
                    ).transposed()
                    // print("Logits:")
                    // print(logits)
                    // print("Probabilities:")
                    // print(probabilities)
                    return softmaxCrossEntropy(
                        logits: logits, //.flattened(),
                        probabilities: probabilities
                    )
                    //sigmoidCrossEntropy(
                    //    logits: labels_.flattened().gathering(atIndices: batch.indices) + 0.001,
                    //    labels: batch.labels + 0.001
                    //)
                }
                optimizer.update(&model, along: grad)
                // print("Loss")
                // print(loss)
                losses.append(loss.replaceNans(with: Tensor(0)).scalarized())
            }
            // print("Losses")
            // print(losses)
            let aggregatedLoss = losses.reduce(0, +) / Float(losses.count)
            if (aggregatedLoss == 0) && isFirstEpoch {
                throw InitializationError.unsuccessfulInitialization(message: "Unlucky weights selection")
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(aggregatedLoss)")
            if isFirstEpoch {
                isFirstEpoch = false
            }
        }
    }
}
