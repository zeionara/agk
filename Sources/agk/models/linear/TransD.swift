import Foundation
import TensorFlow
import Checkpoints

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

private let MODEL_NAME = "transd"

private let ENTITY_EMBEDDINGS_TENSOR_KEY = "entity-embeddings"
private let RELATIONSHIP_EMBEDDINGS_TENSOR_KEY = "relationship-embeddings"
private let ENTITY_PROJECTORS_TENSOR_KEY = "entity-projectors"
private let RELATIONSHIP_PROJECTORS_TENSOR_KEY = "relationship-projectors"

public struct TransD<SourceElement, NormalizedElement>: LinearGraphModel, EntityEmbedder where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public var entityEmbeddings: Embedding<Float>
    public var relationshipEmbeddings: Embedding<Float>
    public var relationshipProjectors: Embedding<Float>
    public var entityProjectors: Embedding<Float>
    @noDerivative public let device: Device
    @noDerivative public let dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>?

    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>? = Optional.none, device device_: Device = Device.default,
                entityEmbeddings: Embedding<Float>? = Optional.none, relationshipEmbeddings: Embedding<Float>? = Optional.none,
                relationshipProjectors: Embedding<Float>? = Optional.none, entityProjectors: Embedding<Float>? = Optional.none) {
        self.entityEmbeddings = entityEmbeddings ?? initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.entities.count, device: device_)
        self.relationshipEmbeddings = relationshipEmbeddings ?? initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        self.relationshipProjectors = relationshipProjectors ?? initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        self.entityProjectors = entityProjectors ?? initEmbeddings(dimensionality: embeddingDimensionality, nItems: dataset!.frame.relationships.count, device: device_)
        device = device_
        self.dataset = dataset
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
        self.entityProjectors = Embedding(
            copying: Embedding(
                embeddings: Tensor<Float>(checkpointOperator.loadTensor(named: ENTITY_PROJECTORS_TENSOR_KEY))
            ), to: device
        )
        self.relationshipProjectors = Embedding(
            copying: Embedding(
                embeddings: Tensor<Float>(checkpointOperator.loadTensor(named: RELATIONSHIP_PROJECTORS_TENSOR_KEY))
            ), to: device
        )
        self.device = device
    }

    public func normalizeEmbeddings() -> TransD {
        TransD(
                embeddingDimensionality: relationshipEmbeddings.embeddings.shape.last!,
                dataset: dataset,
                device: device,
                entityEmbeddings: Embedding(embeddings: normalizeWithL2(tensor: entityEmbeddings.embeddings)),
                relationshipEmbeddings: relationshipEmbeddings,
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

    public func save() throws {
        try CheckpointWriter(
            tensors: [
                    ENTITY_EMBEDDINGS_TENSOR_KEY: entityEmbeddings.embeddings,
                    RELATIONSHIP_EMBEDDINGS_TENSOR_KEY: relationshipEmbeddings.embeddings,
                    ENTITY_PROJECTORS_TENSOR_KEY: entityProjectors.embeddings,
                    RELATIONSHIP_PROJECTORS_TENSOR_KEY: relationshipProjectors.embeddings
                ]
        ).write(
            to: getModelsCacheRoot(),
            name: "\(MODEL_NAME)-\(dataset?.name ?? "none")"
        )
    }
}
