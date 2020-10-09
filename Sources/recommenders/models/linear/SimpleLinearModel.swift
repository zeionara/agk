import Foundation
import TensorFlow

public protocol SimpleLinearModel: LinearGraphModel {
    var entityEmbeddings: Embedding<Float> { get set }
    var relationshipEmbeddings: Embedding<Float> { get set }
    var device: Device { get }
}

extension SimpleLinearModel {
    public init(embeddingDimensionality: Int = 100, dataset: KnowledgeGraphDataset? = Optional.none, device device_: Device = Device.default) {
        self.init(embeddingDimensionality: embeddingDimensionality, dataset: dataset, device: device_)
    }
}
