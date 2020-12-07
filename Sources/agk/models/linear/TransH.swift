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

private func project(embeddings: Tensor<Float>, projectors: Tensor<Float>) -> Tensor<Float> {
    embeddings - (embeddings * projectors).sum(alongAxes: [-1]) * projectors
}

public struct TransH<SourceElement, NormalizedElement>: LinearGraphModel where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    public var relationshipProjectors: Embedding<Float>
    @noDerivative
    public let device: Device

    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>? = Optional.none, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none, relationshipEmbeddings: Embedding<Float>? = Optional.none,
                relationshipProjectors: Embedding<Float>? = Optional.none) {
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
        device = device_
    }

    public func normalizeEmbeddings() -> TransH {
        TransH(
                embeddingDimensionality: relationshipEmbeddings.embeddings.shape.last!,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2(tensor: entityEmbeddings.embeddings)),
                relationshipEmbeddings: relationshipEmbeddings, // Embedding(embeddings: normalize(tensor: relationshipEmbeddings.embeddings))
                relationshipProjectors: relationshipProjectors
        )
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let projectors = relationshipProjectors(triples.transposed()[2])
        let headEmbeddings = project(embeddings: entityEmbeddings(triples.transposed()[0]), projectors: projectors)
        let tailEmbeddings = project(embeddings: entityEmbeddings(triples.transposed()[1]), projectors: projectors)
        let relationshipEmbeddings_ = relationshipEmbeddings(triples.transposed()[2])
        let score = computeScore(head: headEmbeddings, tail: tailEmbeddings, relationship: relationshipEmbeddings_)
        return score
    }
}


