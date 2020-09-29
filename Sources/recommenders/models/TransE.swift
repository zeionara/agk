import Foundation
import TensorFlow

public struct Triple {
    public let head: Int
    public let relationship: Int
    public let tail: Int
}

public func norm(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(squeezingAxes: [1]))
}

private func initEmbeddings(dimensionality: Int, nItems: Int, device device_: Device) -> Embedding<Float> {
    Embedding(
            embeddings: Tensor<Float>(
                    randomUniform: [nItems, dimensionality],
                    lowerBound: Tensor(Float(-1.0) / Float(dimensionality), on: device_),
                    upperBound: Tensor(Float(1.0) / Float(dimensionality), on: device_),
                    on: device_
            )
    )
}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
    let normalizedHead = head.batchNormalized(alongAxis: 1)
    let normalizedTail = tail.batchNormalized(alongAxis: 1)
    let normalizedRelationship = relationship.batchNormalized(alongAxis: 1)

    let score = normalizedHead + (normalizedRelationship - normalizedTail)
    let norma = norm(data: score)

    return norma
}


public struct TransE: GraphModel {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    @noDerivative
    public let device: Device


    public init(entityEmbeddingDimensionality: Int = 100, relationshipEmbeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset, device device_: Device = Device.default) {
        entityEmbeddings = initEmbeddings(dimensionality: entityEmbeddingDimensionality, nItems: dataset.frame.entities.count, device: device_)
        relationshipEmbeddings = initEmbeddings(dimensionality: relationshipEmbeddingDimensionality, nItems: dataset.frame.relationships.count, device: device_)
        device = device_
    }


    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let headEmbeddings = entityEmbeddings(triples.transposed()[0])
        let tailEmbeddings = entityEmbeddings(triples.transposed()[1])
        let relationshipEmbeddings_ = relationshipEmbeddings(triples.transposed()[2])
        let score = computeScore(head: headEmbeddings, tail: tailEmbeddings, relationship: relationshipEmbeddings_)
        return score
    }
}
