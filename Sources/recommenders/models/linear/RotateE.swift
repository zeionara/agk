import Foundation
import TensorFlow

typealias ComplexNumber = (real: Tensor<Float>, imaginary: Tensor<Float>)

private func computeL2Norm_(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(alongAxes: [1]))
}

private func normalizeWithL2_(tensor: Tensor<Float>) -> Tensor<Float> {
    tensor / computeL2Norm_(data: tensor)
}

private func computeScore(head: ComplexNumber, tail: ComplexNumber, relationship: ComplexNumber) -> Tensor<Float> {
    let complexScore: ComplexNumber = (
            real: (relationship.real * tail.real + relationship.imaginary * tail.imaginary) - head.real,
            imaginary: (relationship.real * tail.imaginary - relationship.imaginary * tail.real) - head.imaginary
    )
    let score = Tensor<Float>(stacking: [complexScore.real, complexScore.imaginary]).reshaped(to: [relationship.real.shape[0], -1])
    return normalizeWithL2_(tensor: score).sum(alongAxes: 1).flattened()
}

private func asComplexNumbers(embeddings: Tensor<Float>) -> ComplexNumber {
    let parts = embeddings.split(sizes: Array(repeating: embeddings.shape[1] / 2, count: 2), alongAxis: 1)
    return (real: parts[0], imaginary: parts[1])
}

private func asComplexRotations(embeddings: Tensor<Float>) -> ComplexNumber {
    (real: cos(embeddings), imaginary: sin(embeddings))
}

public struct RotatE: SimpleLinearModel {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    @noDerivative
    public let device: Device

    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset? = Optional.none, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none, relationshipEmbeddings: Embedding<Float>? = Optional.none) {
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality * 2, nItems: dataset!.frame.entities.count, device: device_)
        }
        if let relationshipEmbeddings_ = relationshipEmbeddings {
            self.relationshipEmbeddings = relationshipEmbeddings_
        } else {
            self.relationshipEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        }
        device = device_
    }

    public func normalizeEmbeddings() -> RotatE {
        RotatE(
                embeddingDimensionality: relationshipEmbeddings.embeddings.shape.last!,
                device: device,
                entityEmbeddings: entityEmbeddings, // Embedding(embeddings: normalizeWithL2_(tensor: entityEmbeddings.embeddings)),
                relationshipEmbeddings: Embedding(embeddings: normalizeWithL2_(tensor: relationshipEmbeddings.embeddings))
        )
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let headEmbeddings = asComplexNumbers(embeddings: entityEmbeddings(triples.transposed()[0]))
        let tailEmbeddings = asComplexNumbers(embeddings: entityEmbeddings(triples.transposed()[1]))
        let relationshipEmbeddings_ = asComplexRotations(embeddings: relationshipEmbeddings(triples.transposed()[2]))
        let score = computeScore(head: headEmbeddings, tail: tailEmbeddings, relationship: relationshipEmbeddings_)
        return score
    }
}
