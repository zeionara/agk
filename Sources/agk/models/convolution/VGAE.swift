import Foundation
import TensorFlow
import Checkpoints

private func computeL2Norm_(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(alongAxes: [1]))
}

private func normalizeWithL2_(tensor: Tensor<Float>) -> Tensor<Float> {
    tensor / computeL2Norm_(data: tensor)
}

let ENTITY_EMBEDDINGS_TENSOR_KEY = "entity-embeddings"
let INPUT_LAYER_TENSOR_KEY = "input-layer-tensor-key"

public struct VGAE<SourceElement, NormalizedElement>: ConvolutionGraphModel where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    // public var outputLayer: Tensor<Float>
    private var inputLayer: Dense<Float>
    // private var hiddenLayer: Dense<Float>
    @noDerivative public let device: Device
    @noDerivative public let dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>?


    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>? = Optional.none, device device_: Device = Device.default,
                hiddenLayerSize: Int = 10,
                activation: @escaping Dense<Float>.Activation = relu,
                entityEmbeddings: Embedding<Float>? = Optional.none,
                inputLayer: Dense<Float>? = Optional.none
                // hiddenLayer: Dense<Float>? = Optional.none,
                // outputLayer: Tensor<Float>? = Optional.none
                ) {
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            let nEntities = dataset!.frame.entities.count + dataset!.frame.relationships.count * 2
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: nEntities, device: device_)
        }
        self.inputLayer = inputLayer ?? Dense<Float>(copying: Dense<Float>(inputSize: embeddingDimensionality, outputSize: hiddenLayerSize, activation: activation), to: device_)
        self.dataset = dataset
        // if let inputLayer_ = inputLayer {
        //     self.inputLayer = inputLayer_
        // } else {
        //     self.inputLayer = Dense<Float>(copying: Dense<Float>(inputSize: embeddingDimensionality, outputSize: hiddenLayerSize, activation: activation), to: device_)
        // }
        // if let hiddenLayer_ = hiddenLayer {
        //     self.hiddenLayer = hiddenLayer_
        // } else {
        //     self.hiddenLayer = Dense<Float>(copying: Dense<Float>(inputSize: hiddenLayerSize, outputSize: hiddenLayerSize, activation: activation), to: device_)
        // }
        // if let outputLayer_ = outputLayer {
        //     self.outputLayer = outputLayer_
        // } else {
        //     self.outputLayer = Tensor<Float>(
        //             randomUniform: [hiddenLayerSize, 1],
        //             lowerBound: Tensor(Float(-1.0) / Float(hiddenLayerSize), on: device_),
        //             upperBound: Tensor(Float(1.0) / Float(hiddenLayerSize), on: device_),
        //             on: device_
        //     )
        // }
        device = device_
    }

    public init(
        dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>,
        device: Device = Device.default,
        activation: @escaping Dense<Float>.Activation = relu
                // entityEmbeddings: Embedding<Float>? = Optional.none,
                // inputLayer: Dense<Float>? = Optional.none
                // hiddenLayer: Dense<Float>? = Optional.none,
                // outputLayer: Tensor<Float>? = Optional.none
    ) throws {
         let checkpointOperator = try CheckpointReader(
            checkpointLocation: getModelsCacheRoot().appendingPathComponent("vgae-\(dataset.name)"),
            modelName: "vgae-\(dataset.name)"
        )

        self.dataset = dataset
        self.entityEmbeddings = Embedding(
            copying: Embedding(
                embeddings: Tensor<Float>(checkpointOperator.loadTensor(named: ENTITY_EMBEDDINGS_TENSOR_KEY))
            ), to: device
        )
        self.inputLayer = Dense<Float>(
            copying: Dense<Float>(
            weight: Tensor<Float>(checkpointOperator.loadTensor(named: INPUT_LAYER_TENSOR_KEY)),
                activation: activation
            ),
            to: device
        )
        self.device = device
    }

    public func normalizeEmbeddings() -> VGAE {
//        return self
        return VGAE(
                embeddingDimensionality: entityEmbeddings.embeddings.shape[1],
                dataset: dataset,
                device: device,
                hiddenLayerSize: inputLayer.weight.shape[1],
                // activation: hiddenLayer.activation,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2_(tensor: entityEmbeddings.embeddings)),
                inputLayer: Dense<Float>(weight: normalizeWithL2_(tensor: inputLayer.weight), bias: inputLayer.bias, activation: inputLayer.activation)
                // hiddenLayer: Dense<Float>(weight: normalizeWithL2_(tensor: hiddenLayer.weight), bias: hiddenLayer.bias, activation: hiddenLayer.activation),
                // outputLayer: normalizeWithL2_(tensor: outputLayer)
        )
    }

    @differentiable
    public func callAsFunction(_ matrix: Tensor<Float>) -> Tensor<Float> {
        // let tunedDegreeMatrix = sqrt(Tensor<Float>(matrix.degree)).inverse
        // let tunedMatrix = matmul(matmul(tunedDegreeMatrix, Tensor<Float>(matrix)), tunedDegreeMatrix)
        let output: Tensor<Float> = inputLayer(matmul(matrix, entityEmbeddings.embeddings))
        return sigmoid(matmul(output, output.transposed()))
    }

    public func save() throws {
        try CheckpointWriter(
            tensors: [
                    ENTITY_EMBEDDINGS_TENSOR_KEY: entityEmbeddings.embeddings,
                    INPUT_LAYER_TENSOR_KEY: inputLayer.weight
                ]
        ).write(
            to: getModelsCacheRoot(),
            name: "vgae-\(dataset?.name ?? "none")"
        )
    }
}
