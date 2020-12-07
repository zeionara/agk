import Foundation
import TensorFlow

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

public struct ConvE<SourceElement, NormalizedElement>: ConvolutionGraphModel where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    public var convolutionFilters: DepthwiseConv2D<Float>
    @noDerivative
    public let device: Device
    public let stackedEmbeddingsWidth: Int
    public let stackedEmbeddingsHeight: Int
    public let denseLayer: Dense<Float>

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
        device = device_
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
}
