import Foundation
import TensorFlow

//private func computeL2Norm(data: Tensor<Float>) -> Tensor<Float> {
//    sqrt((data * data).sum(alongAxes: [1]))
//}
//
//private func normalizeWithL2(tensor: Tensor<Float>) -> Tensor<Float> {
//    tensor / computeL2Norm(data: tensor)
//}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
//    let normalizedHead = normalizeWithL2(tensor: head)
//    let normalizedTail = normalizeWithL2(tensor: tail)
//    let normalizedRelationship = normalizeWithL2(tensor: relationship)
    let score = head + (relationship - tail)
    let norma = computeL2Norm(data: score)

    return norma
}

public struct GCN: GraphModel {
    public var entityEmbeddings: Embedding<Float>
    @noDerivative
    public let device: Device


    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset? = Optional.none, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none) {
        if let entityEmbeddings_ = entityEmbeddings {
            self.entityEmbeddings = entityEmbeddings_
        } else {
            self.entityEmbeddings = initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.entities.count, device: device_)
        }
        device = device_
    }

    public func normalizeEmbeddings() -> GCN {
        GCN(
                embeddingDimensionality: 100,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2(tensor: entityEmbeddings.embeddings))
        )
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
//        let headEmbeddings = entityEmbeddings(triples.transposed()[0])
//        let tailEmbeddings = entityEmbeddings(triples.transposed()[1])
//        let relationshipEmbeddings_ = relationshipEmbeddings(triples.transposed()[2])
//        let score = computeScore(head: headEmbeddings, tail: tailEmbeddings, relationship: relationshipEmbeddings_)
        return Tensor<Float>(0.0)
    }
}
