import Foundation
import TensorFlow
import Checkpoints

public struct ConvolutionClassificationTrainer {
    public let nEpochs: Int
    public let batchSize: Int

    public init(nEpochs: Int, batchSize: Int) {
        self.nEpochs = nEpochs
        self.batchSize = batchSize
    }

    public func train<Model, SourceElement>(dataset: KnowledgeGraphDataset<SourceElement, Int32>, model: inout Model, optimizer: Adam<Model>) where Model: ConvolutionGraphModel, Model.Scalar == Float {
        for i in 1...nEpochs{
            var losses: [Float] = []
            for batch in dataset.frame.batched(size: batchSize) {
                print("Generated batch")

                var tunedMatrix: Tensor<Float>
                if tunedDegreeMatrices == Optional.none {
                    print("1")
                    let tunedDegreeMatrix = sqrt(Tensor<Float>(batch.adjacencyTensor.degree)).inverse
                    print("2")
                    let tunedMatrix_ = matmul(matmul(tunedDegreeMatrix, Tensor<Float>(batch.adjacencyTensor)), tunedDegreeMatrix)
                    print("3")
                    tunedDegreeMatrices = tunedMatrix_
                    tunedMatrix = tunedMatrix_
                    do {
                        try CheckpointWriter(tensors: ["matrix": tunedDegreeMatrices!]).write(to: URL(fileURLWithPath: "/home/zeio/agk/data/gcn.agk"), name: "matrix")
                    } catch let exception {
                        print(exception)
                    }
                } else {
                    tunedMatrix = tunedDegreeMatrices!
                }
                
                let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
                    print("Computing model labels")
                    let labels_ = model(tunedMatrix) // model(Tensor<Int32>(batch.adjacencyTensor))
                    print("Computing loss")
                    // print(labels_.flattened().gathering(atIndices: dataset.labelFrame!.indices))
                    // print(dataset.labelFrame!.labels)
                    return sigmoidCrossEntropy(logits: labels_.flattened().gathering(atIndices: dataset.labelFrame!.indices) + 0.001, labels: dataset.labelFrame!.labels + 0.001)
                    // kullbackLeiblerDivergence(predicted: labels_.flattened().gathering(atIndices: dataset.labelFrame!.indices) + 0.001, expected: dataset.labelFrame!.labels + 0.001)
                }
                // print(grad)
                // print(grad)
                optimizer.update(&model, along: grad)
                losses.append(loss.scalarized())
            }
            print("\(i) / \(nEpochs) Epoch. Loss: \(losses.reduce(0, +) / Float(losses.count))")
        }
    }
}