import Foundation
import TensorFlow


public struct GCN: GraphModel {
    public var entityEmbeddings: Embedding<Float>
    public var outputLayer: Tensor<Float>
    private var inputLayer: Dense<Float>
    private var hiddenLayers: Sequential3<Dense<Float>, Dense<Float>, Dense<Float>>
    @noDerivative
    public let device: Device


    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset? = Optional.none, device device_: Device = Device.default, entityEmbeddings: Embedding<Float>? = Optional.none,
                hiddenLayerSize: Int = 10, activation: @escaping Dense<Float>.Activation = relu) {
        let nEntities = dataset!.frame.entities.count + dataset!.frame.relationships.count * 2
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: nEntities, device: device_)
        }
        outputLayer = Tensor<Float>(
                randomUniform: [hiddenLayerSize, 1],
                lowerBound: Tensor(Float(-1.0) / Float(hiddenLayerSize), on: device_),
                upperBound: Tensor(Float(1.0) / Float(hiddenLayerSize), on: device_),
                on: device_
        )
        inputLayer = Dense(inputSize: embeddingDimensionality, outputSize: hiddenLayerSize, activation: activation)
        hiddenLayers = Sequential {
            Dense<Float>(inputSize: hiddenLayerSize, outputSize: hiddenLayerSize, activation: activation)
            Dense<Float>(inputSize: hiddenLayerSize, outputSize: hiddenLayerSize, activation: activation)
            Dense<Float>(inputSize: hiddenLayerSize, outputSize: hiddenLayerSize, activation: activation)
        }
        device = device_
    }

    public func normalizeEmbeddings() -> GCN {
        GCN(
                embeddingDimensionality: 100,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2(tensor: entityEmbeddings.embeddings))
        )
    }

    @differentiable
    public func callAsFunction(_ matrix: Tensor<Int32>) -> Tensor<Float> {
//        print(matrix)
//        let exp = sqrt(Tensor<Float>(matrix.degree))
//        var exp = matrix
//        let i = exp.shape[0] - 1
//        let j = exp.shape[1] - 1
//        print()
//        print(matrix.getMinor(withoutRow: 0, withoutColumn: 0))
        let tunedDegreeMatrix = sqrt(Tensor<Float>(matrix.degree)).inverse
        let tunedMatrix = matmul(matmul(tunedDegreeMatrix, Tensor<Float>(matrix)), tunedDegreeMatrix)
//        let unstackedWeights = weights.unstacked(alongAxis: 2)
//        print(unstackedWeights[0].shape)
//        print(matmul(matmul(tunedMatrix, entityEmbeddings.embeddings), weights).unstacked(alongAxis: 2)[0].transposed())
//        var output: Tensor<Float> = matmul(matmul(tunedMatrix, entityEmbeddings.embeddings), weights).unstacked(alongAxis: 2)[0].transposed()
        var output: Tensor<Float> = inputLayer(matmul(tunedMatrix, entityEmbeddings.embeddings))
//        for hiddenLayer in hiddenLayers {
        output = hiddenLayers(matmul(tunedMatrix, output))
//        }
//        for hiddenLayerWeights in unstackedWeights {
//        output = matmul(matmul(tunedMatrix, output), unstackedWeights[1])
//        output = unstackedWeights.reduce(output) { result, item in matmul(matmul(tunedMatrix, result), item) }
//        output = matmul(matmul(tunedMatrix, output), unstackedWeights[1])
//        }
//        print(output.shape)
        return softmax(matmul(output, outputLayer).flattened())
    }
}
