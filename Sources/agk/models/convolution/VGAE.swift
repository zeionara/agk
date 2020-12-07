import Foundation
import TensorFlow

private func computeL2Norm_(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(alongAxes: [1]))
}

private func normalizeWithL2_(tensor: Tensor<Float>) -> Tensor<Float> {
    tensor / computeL2Norm_(data: tensor)
}

public struct VGAE<SourceElement, NormalizedElement>: ConvolutionGraphModel where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    public var outputLayer: Tensor<Float>
    private var inputLayer: Dense<Float>
    private var hiddenLayer: Dense<Float>
    @noDerivative
    public let device: Device

    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>? = Optional.none, device device_: Device = Device.default,
                hiddenLayerSize: Int = 10, activation: @escaping Dense<Float>.Activation = relu,
                entityEmbeddings: Embedding<Float>? = Optional.none, inputLayer: Dense<Float>? = Optional.none,
                hiddenLayer: Dense<Float>? = Optional.none, outputLayer: Tensor<Float>? = Optional.none) {
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            let nEntities = dataset!.frame.entities.count + dataset!.frame.relationships.count * 2
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: nEntities, device: device_)
        }
        if let inputLayer_ = inputLayer {
            self.inputLayer = inputLayer_
        } else {
            self.inputLayer = Dense<Float>(copying: Dense<Float>(inputSize: embeddingDimensionality, outputSize: hiddenLayerSize, activation: activation), to: device_)
        }
        if let hiddenLayer_ = hiddenLayer {
            self.hiddenLayer = hiddenLayer_
        } else {
            self.hiddenLayer = Dense<Float>(copying: Dense<Float>(inputSize: hiddenLayerSize, outputSize: hiddenLayerSize, activation: activation), to: device_)
        }
        if let outputLayer_ = outputLayer {
            self.outputLayer = outputLayer_
        } else {
            self.outputLayer = Tensor<Float>(
                    randomUniform: [hiddenLayerSize, 1],
                    lowerBound: Tensor(Float(-1.0) / Float(hiddenLayerSize), on: device_),
                    upperBound: Tensor(Float(1.0) / Float(hiddenLayerSize), on: device_),
                    on: device_
            )
        }
        device = device_
    }

    public func normalizeEmbeddings() -> VGAE {
//        return self
        return VGAE(
                embeddingDimensionality: entityEmbeddings.embeddings.shape[1],
                device: device,
                hiddenLayerSize: hiddenLayer.weight.shape[1],
                activation: hiddenLayer.activation,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2_(tensor: entityEmbeddings.embeddings)),
                inputLayer: Dense<Float>(weight: normalizeWithL2_(tensor: inputLayer.weight), bias: inputLayer.bias, activation: inputLayer.activation),
                hiddenLayer: Dense<Float>(weight: normalizeWithL2_(tensor: hiddenLayer.weight), bias: hiddenLayer.bias, activation: hiddenLayer.activation),
                outputLayer: normalizeWithL2_(tensor: outputLayer)
        )
    }

    @differentiable
    public func callAsFunction(_ matrix: Tensor<Int32>) -> Tensor<Float> {
        let tunedDegreeMatrix = sqrt(Tensor<Float>(matrix.degree)).inverse
        let tunedMatrix = matmul(matmul(tunedDegreeMatrix, Tensor<Float>(matrix)), tunedDegreeMatrix)
        var output: Tensor<Float> = inputLayer(matmul(tunedMatrix, entityEmbeddings.embeddings))
        return matmul(output, output.transposed())
    }
}
