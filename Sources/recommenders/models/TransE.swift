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
//                    on: Device.defaultXLA
            )
    )
}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
//    let normalizedHead = head.batchNormalized(alongAxis: 0)
//    let normalizedTail = tail.batchNormalized(alongAxis: 0)
//    let normalizedRelationship = relationship.batchNormalized(alongAxis: 0)
//
//    let score = normalizedHead + (normalizedRelationship - normalizedTail)
//    let norma = norm(data: score)

    return Tensor<Float>(0.0)
}

private func makeEmbeddings(allEmbeddings: Embedding<Float>, id2Index: [Int: Int], triples: Tensor<Float>, column: Int) -> Tensor<Float> {
    var embeddings: [Tensor<Float>] = []
    let items = Tensor<Int32>(triples.transposed()[column])
    for i in 0...items.shape[0] - 1{
        embeddings.append(
                allEmbeddings(
                        Tensor(
                                Int32(
                                        id2Index[Int(items[i].scalar!)]!
                                )
                        )
                )
        )
    }
    return Tensor(embeddings)
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
        relationshipEmbeddings = initEmbeddings(dimensionality: relationshipEmbeddingDimensionality, nItems: dataset.relationshipsUnique.count)

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
                uniqueKeysWithValues: zip(dataset.relationshipsUnique.map {
                    Int($0)
                }, 0...dataset.relationshipsUnique.count - 1)
        )
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Float>) -> Tensor<Float> {
        let headEmbeddings_ = makeEmbeddings(allEmbeddings: headEmbeddings, id2Index: headId2EmbeddingIndex, triples: triples, column: 0)
        let tailEmbeddings_ = makeEmbeddings(allEmbeddings: tailEmbeddings, id2Index: tailId2EmbeddingIndex, triples: triples, column: 1)
        let relationshipEmbeddings_ = makeEmbeddings(allEmbeddings: relationshipEmbeddings, id2Index: relationshipId2EmbeddingIndex, triples: triples, column: 2)
        let score = computeScore(head: headEmbeddings_, tail: tailEmbeddings_, relationship: relationshipEmbeddings_)
        return score
    }
}
