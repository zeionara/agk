import TensorFlow

// public func computeSumLoss(_ positiveScores: Tensor<Float>, _ negativeScores: Tensor<Float>, _ margin: Float = 2.0) -> Tensor<Float> {
//     max(0, margin + positiveScores.sum() - negativeScores.sum())
// }

public struct QuantumTrainer<SourceElement, NormalizedElement>: Trainer where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    func train(
        model: inout QRescal<SourceElement, NormalizedElement>, lr: Float = 0.03, frame: TripleFrame<Int32>,
        loss: @differentiable (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float> = computeSumLoss, margin: Float = 2.0
    ) {
        func handleBatch(batch: Tensor<Int32>, targetLabels: Tensor<Float>) {
            let inferredLabels = model(batch)
            for layer in 0...1 {
                for qubit in 0..<model.relationshipEmbeddings[0][layer].count {
                    let losses = model.computeLosses(
                        triples: batch, loss: targetLabels - inferredLabels, layer: layer, qubit: qubit
                    )
                    model.updateParams(
                        triples: batch, loss: losses, lr: Double(lr), layer: layer, qubit: qubit
                    )
                }
            }
        }

        // let positiveInferredLabels = model(positiveSamples)
        // print("Inferred label before training (+) \(positiveInferredLabels)")
        // let negativeInferredLabels = model(negativeSamples)
        // print("Inferred label before training (-) \(negativeInferredLabels)")
        
        for i in 1...nEpochs {
            print("Running \(i) epoch...")

            for batch in frame.batched(size: batchSize) {
                let negativeFrame = batch.sampleNegativeFrame(negativeFrame: frame.negative)

            
                let positiveInferredLabels = model(batch.tensor)
                // print("Inferred label (+) \(positiveInferredLabels)")
                let negativeInferredLabels = model(negativeFrame.tensor)
                // print("Inferred label (-) \(negativeInferredLabels)")
                
                // handleBatch(batch: positiveSamples, targetLabels: Tensor<Float>(Array.init(repeating: 1, count: positiveSamples.shape[0])))
                // handleBatch(batch: negativeSamples, targetLabels: Tensor<Float>(Array.init(repeating: 0, count: positiveSamples.shape[0])))
                
                // let inferredLabels = model(triples: positiveSamples)
                // let inferredLabels = model(triples: negativeSamples)
                for layer in 0...1 {
                    for qubit in 0..<model.relationshipEmbeddings[0][layer].count {
                        let losses = model.computeLosses(
                            triples: batch.tensor, loss: Tensor<Float>(
                                Array(
                                    repeating: loss(positiveInferredLabels, negativeInferredLabels, margin),
                                    count: batch.tensor.shape[0]
                                )
                            ), layer: layer, qubit: qubit
                        )
                        model.updateParams(
                            triples: batch.tensor, loss: losses, lr: Double(lr), layer: layer, qubit: qubit
                        )
                    }
                }
            }
        }

        // let _positiveInferredLabels = model(positiveSamples)
        // print("Inferred label after training (+) \(_positiveInferredLabels)")
        // let _negativeInferredLabels = model(negativeSamples)
        // print("Inferred label after training(-) \(_negativeInferredLabels)")
    }

}