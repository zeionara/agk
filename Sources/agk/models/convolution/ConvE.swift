import Foundation
import TensorFlow
import Checkpoints

//typealias ComplexNumber = (real: Tensor<Float>, imaginary: Tensor<Float>)
//
//private func computeL2Norm_(data: Tensor<Float>) -> Tensor<Float> {
//    sqrt((data * data).sum(alongAxes: [1]))
//}
//
//private func normalizeWithL2_(tensor: Tensor<Float>) -> Tensor<Float> {
//    tensor / computeL2Norm_(data: tensor)
//}
//
//private func computeScore(head: ComplexNumber, tail: ComplexNumber, relationship: ComplexNumber) -> Tensor<Float> {
//    let complexScore: ComplexNumber = (
//            real: (relationship.real * tail.real + relationship.imaginary * tail.imaginary) - head.real,
//            imaginary: (relationship.real * tail.imaginary - relationship.imaginary * tail.real) - head.imaginary
//    )
//    let score = Tensor<Float>(stacking: [complexScore.real, complexScore.imaginary]).reshaped(to: [relationship.real.shape[0], -1])
//    return normalizeWithL2_(tensor: score).sum(alongAxes: 1).flattened()
//}
//
//private func asComplexNumbers(embeddings: Tensor<Float>) -> ComplexNumber {
//    let parts = embeddings.split(sizes: Array(repeating: embeddings.shape[1] / 2, count: 2), alongAxis: 1)
//    return (real: parts[0], imaginary: parts[1])
//}
//
//private func asComplexRotations(embeddings: Tensor<Float>) -> ComplexNumber {
//    (real: cos(embeddings), imaginary: sin(embeddings))
//}

private let MODEL_NAME = "conve"

private let ENTITY_EMBEDDINGS_TENSOR_KEY = "entity-embeddings"
private let RELATIONSHIP_EMBEDDINGS_TENSOR_KEY = "relationships-embeddings"
private let CONVOLUTION_FILTERS_KEY = "convolution-filters"
private let STACKED_EMBEDDINGS_WIDTH_KEY = "stacked-embeddings-width"
private let STACKED_EMBEDDINGS_HEIGHT_KEY = "stacked-embeddings-height"
private let DENSE_LAYER_KEY = "dense"

public struct ConvE<SourceElement, NormalizedElement>: EntityEmbedder, SaveableGraphModel where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    public var convolutionFilters: DepthwiseConv2D<Float>
    @noDerivative public let device: Device
    @noDerivative public let stackedEmbeddingsWidth: Int
    @noDerivative public let stackedEmbeddingsHeight: Int
    @noDerivative public let dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>?
    public var denseLayer: Dense<Float>

    public init(embeddingDimensionality: Int = 100, stackedEmbeddingsWidth: Int = 25, stackedEmbeddingsHeight: Int = 4, filterWidth: Int = 5, filterHeight: Int = 2,
                nConvolutionalFilters: Int = 3, dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>? = Optional.none, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none, relationshipEmbeddings: Embedding<Float>? = Optional.none,
                convolutionFilters: DepthwiseConv2D<Float>? = Optional.none, activation: @escaping Dense<Float>.Activation = relu,
                denseLayer: Dense<Float>? = Optional.none) {
        assert(stackedEmbeddingsWidth * stackedEmbeddingsHeight == embeddingDimensionality)
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.entities.count, device: device_)
        }
        if let relationshipEmbeddings_ = relationshipEmbeddings {
            self.relationshipEmbeddings = relationshipEmbeddings_
        } else {
            self.relationshipEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        }
        self.convolutionFilters = convolutionFilters ?? DepthwiseConv2D(filterShape: (filterHeight, filterWidth, 1, nConvolutionalFilters), activation: activation)
        self.stackedEmbeddingsWidth = stackedEmbeddingsWidth
        self.stackedEmbeddingsHeight = stackedEmbeddingsHeight
        self.denseLayer = denseLayer ?? Dense<Float>(
                inputSize: ((stackedEmbeddingsHeight * 2 - filterHeight) + 1) * (stackedEmbeddingsWidth - filterWidth + 1) * nConvolutionalFilters,
                outputSize: embeddingDimensionality,
                activation: activation
        )
        self.dataset = dataset
        device = device_
    }

    public init(
        dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>,
        device: Device = Device.default,
        activation: @escaping Dense<Float>.Activation = relu
    ) throws {
         let checkpointOperator = try CheckpointReader(
            checkpointLocation: getModelsCacheRoot().appendingPathComponent("\(MODEL_NAME)-\(dataset.name)"),
            modelName: "\(MODEL_NAME)-\(dataset.name)"
        )

        self.dataset = dataset
        self.entityEmbeddings = Embedding(
            copying: Embedding(
                embeddings: Tensor<Float>(checkpointOperator.loadTensor(named: ENTITY_EMBEDDINGS_TENSOR_KEY))
            ), to: device
        )
        self.relationshipEmbeddings = Embedding(
            copying: Embedding(
                embeddings: Tensor<Float>(checkpointOperator.loadTensor(named: RELATIONSHIP_EMBEDDINGS_TENSOR_KEY))
            ), to: device
        )
        self.convolutionFilters = DepthwiseConv2D(
            copying: DepthwiseConv2D(
                filter: Tensor<Float>(checkpointOperator.loadTensor(named: CONVOLUTION_FILTERS_KEY)),
                activation: activation
            ),
            to: device
        )
        self.stackedEmbeddingsWidth = Int(Tensor<Float>(checkpointOperator.loadTensor(named: STACKED_EMBEDDINGS_WIDTH_KEY)).scalar!)
        self.stackedEmbeddingsHeight = Int(Tensor<Float>(checkpointOperator.loadTensor(named: STACKED_EMBEDDINGS_HEIGHT_KEY)).scalar!)
        self.denseLayer = Dense<Float>(
            copying: Dense<Float>(
            weight: Tensor<Float>(checkpointOperator.loadTensor(named: DENSE_LAYER_KEY)),
                activation: activation
            ),
            to: device
        )
        self.device = device
    }

    public var filterWidth: Int {
        convolutionFilters.filter.shape[0]
    }

    public var filterHeight: Int {
        convolutionFilters.filter.shape[1]
    }

    public var nConvolutionalFilters: Int {
        convolutionFilters.filter.shape[3]
    }

    public func normalizeEmbeddings() -> ConvE {
        ConvE(
                embeddingDimensionality: relationshipEmbeddings.embeddings.shape.last!,
                stackedEmbeddingsWidth: stackedEmbeddingsWidth,
                stackedEmbeddingsHeight: stackedEmbeddingsHeight,
                device: device,
                entityEmbeddings: entityEmbeddings, // Embedding(embeddings: normalizeWithL2_(tensor: entityEmbeddings.embeddings)),
                relationshipEmbeddings: relationshipEmbeddings, // Embedding(embeddings: normalizeWithL2_(tensor: relationshipEmbeddings.embeddings))
                convolutionFilters: convolutionFilters,
                activation: convolutionFilters.activation
        )
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let headEmbeddings = entityEmbeddings(triples.transposed()[0])
        let tailEmbeddings = entityEmbeddings(triples.transposed()[1])
        let relationshipEmbeddings_ = relationshipEmbeddings(triples.transposed()[2])
        var stackedEmbeddings = Tensor(
                stacking: [
                    headEmbeddings.reshaped(to: [-1, convolutionFilters.filter.shape[2], convolutionFilters.filter.shape[3]]),
                    relationshipEmbeddings_.reshaped(to: [-1, convolutionFilters.filter.shape[2], convolutionFilters.filter.shape[3]])
                ], alongAxis: 1
        ).reshaped(to: [headEmbeddings.shape[0], -1, stackedEmbeddingsWidth, 1])
        let convolutionResult = convolutionFilters(stackedEmbeddings)
        let multiplicationResult = denseLayer(convolutionResult.reshaped(to: [convolutionResult.shape[0], -1]))
        return softmax(matmul(multiplicationResult, entityEmbeddings.embeddings.transposed()))
    }

    public func save() throws {
        try CheckpointWriter(
            tensors: [
                    ENTITY_EMBEDDINGS_TENSOR_KEY: entityEmbeddings.embeddings,
                    RELATIONSHIP_EMBEDDINGS_TENSOR_KEY: relationshipEmbeddings.embeddings,
                    CONVOLUTION_FILTERS_KEY: convolutionFilters.filter,
                    STACKED_EMBEDDINGS_HEIGHT_KEY: Tensor<Float>(Float(stackedEmbeddingsHeight)),
                    STACKED_EMBEDDINGS_WIDTH_KEY: Tensor<Float>(Float(stackedEmbeddingsWidth)),
                    DENSE_LAYER_KEY: denseLayer.weight
                ]
        ).write(
            to: getModelsCacheRoot(),
            name: "\(MODEL_NAME)-\(dataset?.name ?? "none")"
        )
    }
}
