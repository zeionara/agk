import Foundation
import TensorFlow
import Checkpoints

var tunedDegreeMatrices: Tensor<Float>? = Optional.none // [Tensor<Int32>: Tensor<Float>]()

private func computeL2Norm_(data: Tensor<Float>, axis: Int = 0) -> Tensor<Float> {
    sqrt((data * data).sum(alongAxes: [axis]))
}

private func normalizeWithL2_(tensor: Tensor<Float>, axis: Int = 0) -> Tensor<Float> {
    tensor / computeL2Norm_(data: tensor, axis: axis)
}

// extension Tensor where Scalar: TensorFlowFloatingPoint {
//     @differentiable(wrt: rhs, )
//     static func multiply (_ lhs: Tensor,  _ rhs: Tensor) -> Tensor {
//         return matmul(lhs, rhs)
//     }
// }

public struct GCN<SourceElement, NormalizedElement>: ConvolutionGraphModel where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    // public var outputLayer: Dense<Float>
    // private var inputLayer: Dense<Float>
    private var dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>
    // private var hiddenLayers: [Dense<Float>]
    @noDerivative
    public let device: Device
    // @noDerivative public var tunedDegreeMatrices: [[Float]: Tensor<Float>]

    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>? = Optional.none, device device_: Device = Device.default,
                hiddenLayerSize: Int = 10, activation: @escaping Dense<Float>.Activation = relu, entityEmbeddings: Embedding<Float>? = Optional.none
                // inputLayer: Dense<Float>? = Optional.none, 
                // outputLayer: Dense<Float>? = Optional.none
                ) {
        let nEntities = dataset!.frame.entities.count + dataset!.frame.relationships.count * 2
        self.dataset = dataset!
        self.entityEmbeddings = entityEmbeddings ?? initEmbeddings(dimensionality: embeddingDimensionality, nItems: nEntities, device: device_)
        // self.inputLayer = inputLayer ?? Dense(inputSize: embeddingDimensionality, outputSize: hiddenLayerSize, activation: activation)
        // hiddenLayers = Array(repeating: Dense<Float>(inputSize: hiddenLayerSize, outputSize: hiddenLayerSize, activation: activation), count: 3)
        var checkpointOperator: CheckpointReader? = Optional.none
        do {
            checkpointOperator = try CheckpointReader(checkpointLocation: URL(fileURLWithPath: "/home/zeio/agk/data/gcn.agk/matrix"), modelName: "matrix")
            // checkpointOperator = CheckpointWriter()
            // print(checkpointOperator?.containsTensor(named: "matrix"))
            // print(checkpointOperator?.containsTensor(named: "matrix/matrix"))
            // print(checkpointOperator?.containsTensor(named: "matrix/matrix"))
            tunedDegreeMatrices = Tensor<Float>(checkpointOperator!.loadTensor(named: "matrix"))
        } catch let exception {
            print(exception)
        }
        // outputLayer = Tensor<Float>(
        //         randomUniform: [hiddenLayerSize, 1],
        //         lowerBound: Tensor(Float(-1.0) / Float(hiddenLayerSize), on: device_),
        //         upperBound: Tensor(Float(1.0) / Float(hiddenLayerSize), on: device_),
        //         on: device_
        // )
        // self.outputLayer = outputLayer ?? Dense<Float>(inputSize: embeddingDimensionality, outputSize: 1, activation: activation)
        device = device_
    }

    public func normalizeEmbeddings() -> GCN {
        return GCN(
            dataset: dataset,
            device: device,
            entityEmbeddings: Embedding<Float>(embeddings: normalizeWithL2_(tensor: entityEmbeddings.embeddings, axis: 1))
            // inputLayer: inputLayer,
            // outputLayer: outputLayer
        )
    }

    @differentiable
    private func passThroughHiddenLayer(tensor: Tensor<Float>, tunedMatrix: Tensor<Float>, layerIndex: Int) -> Tensor<Float> {
        // return matmul(tunedMatrix, hiddenLayers[layerIndex](tensor))
        return tensor
    }

    @differentiable
    private func passThroughHiddenLayers(tensor: Tensor<Float>, tunedMatrix: Tensor<Float>) -> Tensor<Float> {
        // var output = passThroughHiddenLayer(tensor: tensor, tunedMatrix: tunedMatrix, layerIndex: 0)
        // output = passThroughHiddenLayer(tensor: output, tunedMatrix: tunedMatrix, layerIndex: 1)
        // output = passThroughHiddenLayer(tensor: output, tunedMatrix: tunedMatrix, layerIndex: 2)
        // return output
        return tensor
    }

    @differentiable
    public func callAsFunction(_ matrix: Tensor<Float>) -> Tensor<Float> {
        // print(matrix.shape)
        // let flattened = Tensor<Float>(matrix.degree).flattened().map{$0.scalarized()}
        // let flattened = matrix
        var tunedMatrix: Tensor<Float> = normalizeWithL2_(tensor: matrix + 0.001, axis: 1)
        // if tunedDegreeMatrices == Optional.none {
        //     print("1")
        //     let tunedDegreeMatrix = sqrt(Tensor<Float>(matrix.degree)).inverse
        //     print("2")
        //     let tunedMatrix_ = matmul(matmul(tunedDegreeMatrix, Tensor<Float>(matrix)), tunedDegreeMatrix)
        //     print("3")
        //     tunedDegreeMatrices = tunedMatrix_
        //     tunedMatrix = tunedMatrix_
        // } else {
        //     tunedMatrix = tunedDegreeMatrices!
        // }
        // print(tunedMatrix)
        // _Raw.mul(tunedMatrix, entityEmbeddings.embeddings)
        var output: Tensor<Float> = matmul(tunedMatrix, entityEmbeddings.embeddings) // inputLayer(matmul(tunedMatrix, entityEmbeddings.embeddings))
        // print("embeddings (\(entityEmbeddings.embeddings.shape))")
        // // print(entityEmbeddings.embeddings)
        // print("matrix (\(tunedMatrix.shape))")
        // // print(tunedMatrix)
        // print("output")
        // // print(output)
        // print("4")
        // // print(tunedMatrix)
        // output = passThroughHiddenLayers(tensor: output, tunedMatrix: tunedMatrix)
        // print("5")
        // // print(output)
        // // let result = sigmoid(matmul(output, outputLayer).flattened())
        // print("Output shape: \(output.shape)")
        // // print(outputLayer(output).flattened())
        // print(outputLayer(output).flattened())
        // let result = sigmoid(outputLayer(output).flattened())
        // let result = outputLayer(output).flattened()
        // print(result)
        // return result
        return sigmoid(output.sum(alongAxes: [0]))
    }
}
