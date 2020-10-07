import Foundation
import TensorFlow


public struct GCN: ConvolutionGraphModel {
    public var entityEmbeddings: Embedding<Float>
    public var outputLayer: Tensor<Float>
    private var inputLayer: Dense<Float>
    private var hiddenLayers: [Dense<Float>]
    @noDerivative
    public let device: Device

    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset? = Optional.none, device device_: Device = Device.default,
                hiddenLayerSize: Int = 10, activation: @escaping Dense<Float>.Activation = relu) {
        let nEntities = dataset!.frame.entities.count + dataset!.frame.relationships.count * 2
        entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: nEntities, device: device_)
        inputLayer = Dense(inputSize: embeddingDimensionality, outputSize: hiddenLayerSize, activation: activation)
        hiddenLayers = Array(repeating: Dense<Float>(inputSize: hiddenLayerSize, outputSize: hiddenLayerSize, activation: activation), count: 3)
        outputLayer = Tensor<Float>(
                randomUniform: [hiddenLayerSize, 1],
                lowerBound: Tensor(Float(-1.0) / Float(hiddenLayerSize), on: device_),
                upperBound: Tensor(Float(1.0) / Float(hiddenLayerSize), on: device_),
                on: device_
        )
        device = device_
    }

    public func normalizeEmbeddings() -> GCN {
        self
    }

    @differentiable
    private func passThroughHiddenLayer(tensor: Tensor<Float>, tunedMatrix: Tensor<Float>, layerIndex: Int) -> Tensor<Float> {
        matmul(tunedMatrix, hiddenLayers[layerIndex](tensor))
    }

    @differentiable
    private func passThroughHiddenLayers(tensor: Tensor<Float>, tunedMatrix: Tensor<Float>) -> Tensor<Float> {
        var output = passThroughHiddenLayer(tensor: tensor, tunedMatrix: tunedMatrix, layerIndex: 0)
        output = passThroughHiddenLayer(tensor: output, tunedMatrix: tunedMatrix, layerIndex: 1)
        output = passThroughHiddenLayer(tensor: output, tunedMatrix: tunedMatrix, layerIndex: 2)
        return output
    }

    @differentiable
    public func callAsFunction(_ matrix: Tensor<Int32>) -> Tensor<Float> {
        let tunedDegreeMatrix = sqrt(Tensor<Float>(matrix.degree)).inverse
        let tunedMatrix = matmul(matmul(tunedDegreeMatrix, Tensor<Float>(matrix)), tunedDegreeMatrix)
        var output: Tensor<Float> = inputLayer(matmul(tunedMatrix, entityEmbeddings.embeddings))
        output = passThroughHiddenLayers(tensor: output, tunedMatrix: tunedMatrix)
        let result = sigmoid(matmul(output, outputLayer).flattened())
        return result
    }
}
