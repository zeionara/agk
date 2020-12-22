import Foundation
import TensorFlow
import Checkpoints

public func computeL2Norm(data: Tensor<Float>) -> Tensor<Float> {
    sqrt((data * data).sum(alongAxes: [1]))
}

public func normalizeWithL2(tensor: Tensor<Float>) -> Tensor<Float> {
    tensor / computeL2Norm(data: tensor)
}

private func computeScore(head: Tensor<Float>, tail: Tensor<Float>, relationship: Tensor<Float>) -> Tensor<Float> {
    let score = head + (relationship - tail)
    let norma = computeL2Norm(data: score)

    return norma
}

private let MODEL_NAME = "transe"

private let ENTITY_EMBEDDINGS_TENSOR_KEY = "entity-embeddings"
private let RELATIONSHIP_EMBEDDINGS_TENSOR_KEY = "relationship-embeddings"

public struct TransE<SourceElement, NormalizedElement>: LinearGraphModel, EntityEmbedder where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    @noDerivative public let device: Device
    @noDerivative public let dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>?


    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>? = Optional.none, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none, relationshipEmbeddings: Embedding<Float>? = Optional.none) {
        self.dataset = dataset
        self.entityEmbeddings = entityEmbeddings ?? initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.entities.count, device: device_)
        self.relationshipEmbeddings = relationshipEmbeddings ?? initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        device = device_
    }

    public init(
        dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>,
        device: Device = Device.default
    ) throws {
         let checkpointOperator = try CheckpointReader(
            checkpointLocation: getModelsCacheRoot().appendingPathComponent("\(MODEL_NAME)-\(dataset.name)"),
            modelName: "\(MODEL_NAME)-\(dataset.name)"
        )
        self.dataset = dataset
        self.entityEmbeddings = Embedding(
            copying: Embedding(
                embeddings: Tensor<Float>(checkpointOperator.loadTensor(named: ENTITY_EMBEDDINGS_TENSOR_KEY))
            ), to: device
        )
        self.relationshipEmbeddings = Embedding(
            copying: Embedding(
                embeddings: Tensor<Float>(checkpointOperator.loadTensor(named: RELATIONSHIP_EMBEDDINGS_TENSOR_KEY))
            ), to: device
        )
        self.device = device
    }

    public func normalizeEmbeddings() -> TransE {
        TransE(
                embeddingDimensionality: relationshipEmbeddings.embeddings.shape.last!,
                dataset: dataset,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2(tensor: entityEmbeddings.embeddings)),
                relationshipEmbeddings: relationshipEmbeddings // Embedding(embeddings: normalize(tensor: relationshipEmbeddings.embeddings))
        )
    }

    @differentiable
    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let headEmbeddings = entityEmbeddings(triples.transposed()[0])
        // print(headEmbeddings.shape)
        let tailEmbeddings = entityEmbeddings(triples.transposed()[1])
        // print(tailEmbeddings.shape)
        let relationshipEmbeddings_ = relationshipEmbeddings(triples.transposed()[2])
        // print(relationshipEmbeddings_.shape)
        let score = computeScore(head: headEmbeddings, tail: tailEmbeddings, relationship: relationshipEmbeddings_)
        // print(score)
        return score
    }

    public func save() throws {
        try CheckpointWriter(
            tensors: [
                    ENTITY_EMBEDDINGS_TENSOR_KEY: entityEmbeddings.embeddings,
                    RELATIONSHIP_EMBEDDINGS_TENSOR_KEY: relationshipEmbeddings.embeddings
                ]
        ).write(
            to: getModelsCacheRoot(),
            name: "\(MODEL_NAME)-\(dataset?.name ?? "none")"
        )
    }
}
