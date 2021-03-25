import TensorFlow
import Foundation

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
        loss: @differentiable (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float> = computeSumLoss, margin: Float = 2.0,
        enableCompleteLoss: Bool = false, enableAllLayersOptimization: Bool = false
    ) {
        func handleBatch(batch: Tensor<Int32>, targetLabels: Tensor<Float>, i: Optional<Int> = .none) {
            print("Handling \(i ?? 0) batch (\(batch.shape[0]) samples)")
            let inferredLabels = model(batch)
            for layer in 0...1 {
                for qubit in 0..<model.relationshipEmbeddings[0][layer].count {
                    let losses = model.computeLosses(
                        triples: batch, loss: targetLabels - inferredLabels, layer: layer, qubit: qubit, optimizedCircuit: .relationship
                    )
                    model.updateParams(
                        triples: batch, loss: losses, lr: Double(lr), layer: layer, qubit: qubit, optimizedCircuit: .relationship
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

            let batches = frame.batched(size: batchSize)
            let nBatches = batches.count
            for (i, batch) in batches.enumerated() {
                print("Handling \(i) / \(nBatches) batch (\(batch.tensor.shape[0]) samples)")
                let handlingStartTimestamp = DispatchTime.now().uptimeNanoseconds
                let negativeFrame = batch.sampleNegativeFrame(negativeFrame: frame.negative)

            
                let positiveInferredLabels = model(batch.tensor)
                // print("Inferred label (+) \(positiveInferredLabels)")
                let negativeInferredLabels = model(negativeFrame.tensor)
                // print("Inferred label (-) \(negativeInferredLabels)")
                
                // handleBatch(batch: positiveSamples, targetLabels: Tensor<Float>(Array.init(repeating: 1, count: positiveSamples.shape[0])))
                // handleBatch(batch: negativeSamples, targetLabels: Tensor<Float>(Array.init(repeating: 0, count: positiveSamples.shape[0])))
                
                // let inferredLabels = model(triples: positiveSamples)
                // let inferredLabels = model(triples: negativeSamples)
                for layer in (enableAllLayersOptimization ? 0..<model.relationshipEmbeddings[0].count : 0..<2) {
                    // print("Handling layer \(layer)")
                    for qubit in 0..<model.relationshipEmbeddings[0][layer].count {
                        // print("Handling qubit \(qubit)")
                        var losses = model.computeLosses(
                            triples: batch.tensor, loss: Tensor<Float>(
                                Array(
                                    repeating: loss(positiveInferredLabels, negativeInferredLabels, margin),
                                    count: batch.tensor.shape[0]
                                )
                            ), layer: layer, qubit: qubit, optimizedCircuit: .relationship, enableCompleteLoss: enableCompleteLoss
                        )
                        model.updateParams(
                            triples: batch.tensor, loss: losses, lr: Double(lr), layer: layer, qubit: qubit, optimizedCircuit: .relationship
                        )
                        //
                        losses = model.computeLosses(
                            triples: batch.tensor, loss: Tensor<Float>(
                                Array(
                                    repeating: loss(positiveInferredLabels, negativeInferredLabels, margin),
                                    count: batch.tensor.shape[0]
                                )
                            ), layer: layer, qubit: qubit, optimizedCircuit: .subject, enableCompleteLoss: enableCompleteLoss
                        )
                        model.updateParams(
                            triples: batch.tensor, loss: losses, lr: Double(lr), layer: layer, qubit: qubit, optimizedCircuit: .subject
                        )
                        //
                        losses = model.computeLosses(
                            triples: batch.tensor, loss: Tensor<Float>(
                                Array(
                                    repeating: loss(positiveInferredLabels, negativeInferredLabels, margin),
                                    count: batch.tensor.shape[0]
                                )
                            ), layer: layer, qubit: qubit, optimizedCircuit: .object, enableCompleteLoss: enableCompleteLoss
                        )
                        model.updateParams(
                            triples: batch.tensor, loss: losses, lr: Double(lr), layer: layer, qubit: qubit, optimizedCircuit: .object
                        )
                    }
                }
                print("Handled \(i) / \(nBatches) batch in \(Double(DispatchTime.now().uptimeNanoseconds - handlingStartTimestamp) / Double(1_000_000_000)) seconds)")
            }
        }

        // let _positiveInferredLabels = model(positiveSamples)
        // print("Inferred label after training (+) \(_positiveInferredLabels)")
        // let _negativeInferredLabels = model(negativeSamples)
        // print("Inferred label after training(-) \(_negativeInferredLabels)")
    }

}