import Foundation
import TensorFlow


public struct Tester {
    public let batchSize: Int

    public init(batchSize: Int) {
        self.batchSize = batchSize
    }

    public func test<Model, SourceElement>(dataset: KnowledgeGraphDataset<SourceElement, Int32>, model: Model) where Model: GraphModel {
        let trainBatches = dataset.normalizedFrame.batched(size: batchSize)
        for batch in dataset.normalizedNegativeFrame.batched(size: batchSize) {
            let negativeScores = model(batch.tensor)
            let scores = model(trainBatches.randomElement()!.tensor)
            assert(scores.sum().scalarized() < negativeScores.sum().scalarized())
        }
    }
}