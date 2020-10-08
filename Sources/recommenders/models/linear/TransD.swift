import Foundation
import TensorFlow

private func computeL2Norm_(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(alongAxes: [1]))
}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
    let score = head + (relationship - tail)
    let norma = computeL2Norm_(data: score)

    return norma
}

private func project(embeddings: Tensor<Float>, entityProjectors: Tensor<Float>, relationshipProjectors: Tensor<Float>) -> Tensor<Float> {
    embeddings - (embeddings * entityProjectors).sum(alongAxes: [-1]) * relationshipProjectors
}

public struct TransD: LinearGraphModel {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    public var relationshipProjectors: Embedding<Float>
    public var entityProjectors: Embedding<Float>
    @noDerivative
    public let device: Device

    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset? = Optional.none, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none, relationshipEmbeddings: Embedding<Float>? = Optional.none,
                relationshipProjectors: Embedding<Float>? = Optional.none, entityProjectors: Embedding<Float>? = Optional.none) {
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
        if let relationshipProjectors_ = relationshipProjectors {
            self.relationshipProjectors = relationshipProjectors_
        } else {
            self.relationshipProjectors = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        }
        if let entityProjectors_ = entityProjectors {
            self.entityProjectors = entityProjectors_
        } else {
            self.entityProjectors = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        }
        device = device_
    }

    public func normalizeEmbeddings() -> TransD {
        TransD(
                embeddingDimensionality: relationshipEmbeddings.embeddings.shape.last!,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2(tensor: entityEmbeddings.embeddings)),
                relationshipEmbeddings: relationshipEmbeddings, // Embedding(embeddings: normalize(tensor: relationshipEmbeddings.embeddings))
                relationshipProjectors: relationshipProjectors,
                entityProjectors: entityProjectors
        )
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let relationshipProjectors_ = relationshipProjectors(triples.transposed()[2])
        let entityProjectors_ = entityProjectors(triples.transposed()[2])
        let headEmbeddings = project(embeddings: entityEmbeddings(triples.transposed()[0]), entityProjectors: entityProjectors_, relationshipProjectors: relationshipProjectors_)
        let tailEmbeddings = project(embeddings: entityEmbeddings(triples.transposed()[1]), entityProjectors: entityProjectors_, relationshipProjectors: relationshipProjectors_)
        let relationshipEmbeddings_ = relationshipEmbeddings(triples.transposed()[2])
        let score = computeScore(head: headEmbeddings, tail: tailEmbeddings, relationship: relationshipEmbeddings_)
        return score
    }
}
