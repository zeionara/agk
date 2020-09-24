import Foundation
import TensorFlow

public struct Triple {
    public let head: Int
    public let relationship: Int
    public let tail: Int
}

public func norm(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data.flattened() * data.flattened()).sum())
}

private func initEmbeddings(dimensionality: Int, nItems: Int) -> Embedding<Float> {
    Embedding(
            embeddings: Tensor<Float>(
                    randomUniform: [nItems, dimensionality],
                    lowerBound: Tensor(Float(-1.0) / Float(dimensionality)),
                    upperBound: Tensor(Float(1.0) / Float(dimensionality))
            )
    )
}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
    let normalizedHead = head.batchNormalized(alongAxis: 0)
    let normalizedTail = tail.batchNormalized(alongAxis: 0)
    let normalizedRelationship = relationship.batchNormalized(alongAxis: 0)

    let score = normalizedHead + (normalizedRelationship - normalizedTail)
    let norma = norm(data: score)

    return norma
}

struct TransE: Module {
    public var headEmbeddings: Embedding<Float>
    public var tailEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>

    public let headId2EmbeddingIndex: [Int: Int]
    public let tailId2EmbeddingIndex: [Int: Int]
    public let relationshipId2EmbeddingIndex: [Int: Int]


    public init(nodeEmbeddingDimensionality: Int = 100, relationshipEmbeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset) {
        headEmbeddings = initEmbeddings(dimensionality: nodeEmbeddingDimensionality, nItems: dataset.headsUnique.count)
        tailEmbeddings = initEmbeddings(dimensionality: nodeEmbeddingDimensionality, nItems: dataset.tailsUnique.count)
        relationshipEmbeddings = initEmbeddings(dimensionality: relationshipEmbeddingDimensionality, nItems: 1)

        headId2EmbeddingIndex = Dictionary(
                uniqueKeysWithValues: zip(dataset.headsUnique.map {
                    Int($0)
                }, 0...dataset.headsUnique.count - 1)
        )
        tailId2EmbeddingIndex = Dictionary(
                uniqueKeysWithValues: zip(dataset.tailsUnique.map {
                    Int($0)
                }, 0...dataset.tailsUnique.count - 1)
        )
        relationshipId2EmbeddingIndex = Dictionary(
                uniqueKeysWithValues: zip([0], [0])
        )
    }

    @differentiable
    public func callAsFunction(_ triple: Triple) -> Tensor<Float> {
        let headEmbedding = headEmbeddings(Tensor(Int32(headId2EmbeddingIndex[triple.head]!)))
        let tailEmbedding = headEmbeddings(Tensor(Int32(tailId2EmbeddingIndex[triple.tail]!)))
        let relationshipEmbedding = relationshipEmbeddings(Tensor(Int32(relationshipId2EmbeddingIndex[triple.relationship]!)))
        let score = computeScore(head: headEmbedding, tail: tailEmbedding, relationship: relationshipEmbedding)
        return score
    }
}
