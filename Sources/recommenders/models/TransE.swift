import Foundation
import TensorFlow

public struct Triple {
    public let head: Int
    public let relationship: Int
    public let tail: Int
}

private func norm(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(squeezingAxes: [1]))
}

public func initEmbeddings(dimensionality: Int, nItems: Int, device device_: Device) -> Embedding<Float> {
    Embedding(
            embeddings: Tensor<Float>(
                    randomUniform: [nItems, dimensionality],
                    lowerBound: Tensor(Float(-1.0) / Float(dimensionality), on: device_),
                    upperBound: Tensor(Float(1.0) / Float(dimensionality), on: device_),
                    on: device_
            )
    )
}

public func normalize(tensor: Tensor<Float>) -> Tensor<Float> {
    tensor / sqrt((tensor * tensor).sum(alongAxes: [1]))
}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
    let normalizedHead = normalize(tensor: head)
    let normalizedTail = normalize(tensor: tail)
    let normalizedRelationship = normalize(tensor: relationship)

    let score = normalizedHead + (normalizedRelationship - normalizedTail)
    let norma = norm(data: score)

    return norma
}

public struct TransE: GraphModel {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    @noDerivative
    public let device: Device


    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none, relationshipEmbeddings: Embedding<Float>? = Optional.none) {
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset.frame.entities.count, device: device_)
        }
        if let relationshipEmbeddings_ = relationshipEmbeddings {
            self.relationshipEmbeddings = relationshipEmbeddings_
        } else {
            self.relationshipEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset.frame.relationships.count, device: device_)
        }
        device = device_
    }

    public func normalizeEmbeddings<T: GraphModel>() -> T{
        TransE(
                embeddingDimensionality: 100,
                dataset: dataset,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalize(tensor: entityEmbeddings.embeddings)),
                relationshipEmbeddings: Embedding(embeddings: normalize(tensor: relationshipEmbeddings.embeddings))
        ) as! T
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
