import Foundation
import TensorFlow

private func norm(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(squeezingAxes: [1]))
}

private func transfer(embedding: Tensor<Float>, entityOffset: Tensor<Float>, relationshipOffset: Tensor<Float>) -> Tensor<Float> {
    (embedding - (embedding * entityOffset).sum(alongAxes: [-1]) * relationshipOffset).batchNormalized(alongAxis: 1)
}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
    let normalizedHead = head.batchNormalized(alongAxis: 1)
    let normalizedTail = tail.batchNormalized(alongAxis: 1)
    let normalizedRelationship = relationship.batchNormalized(alongAxis: 1)

    let score = normalizedHead + (normalizedRelationship - normalizedTail)
    let norma = norm(data: score)

    return norma
}

public struct TransD: GraphModel {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    public var transferRelationship: Embedding<Float>
    public var transferEntity: Embedding<Float>
    @noDerivative
    public let device: Device


    public init(entityEmbeddingDimensionality: Int = 100, relationshipEmbeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset, device device_: Device = Device.default) {
        entityEmbeddings = initEmbeddings(dimensionality: entityEmbeddingDimensionality, nItems: dataset.frame.entities.count, device: device_)
        relationshipEmbeddings = initEmbeddings(dimensionality: relationshipEmbeddingDimensionality, nItems: dataset.frame.relationships.count, device: device_)
        transferRelationship = initEmbeddings(dimensionality: relationshipEmbeddingDimensionality, nItems: dataset.frame.relationships.count, device: device_)
        transferEntity = initEmbeddings(dimensionality: relationshipEmbeddingDimensionality, nItems: dataset.frame.entities.count, device: device_)
        device = device_
    }

    public func normalizeEmbeddings() {
//        entityEmbeddings = Embedding(embeddings: normalize(tensor: entityEmbeddings.embeddings))
//        relationshipEmbeddings = Embedding(embeddings: normalize(tensor: relationshipEmbeddings.embeddings))
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let transferRelationship_ = transferRelationship(triples.transposed()[2])
        let transferHead = transferEntity(triples.transposed()[0])
        let transferTail = transferEntity(triples.transposed()[1])
        let headEmbeddings = transfer(embedding: entityEmbeddings(triples.transposed()[0]), entityOffset: transferHead, relationshipOffset: transferRelationship_)
        let tailEmbeddings = transfer(embedding: entityEmbeddings(triples.transposed()[1]), entityOffset: transferTail, relationshipOffset: transferRelationship_)
        let relationshipEmbeddings_ = relationshipEmbeddings(triples.transposed()[2])
        let score = computeScore(head: headEmbeddings, tail: tailEmbeddings, relationship: relationshipEmbeddings_)
        return score
    }
}
