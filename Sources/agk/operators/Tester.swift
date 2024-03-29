import Foundation
import TensorFlow


public struct Tester {
    public let batchSize: Int

    public init(batchSize: Int) {
        self.batchSize = batchSize
    }

    public func test<Model, SourceElement>(dataset: KnowledgeGraphDataset<SourceElement, Int32>, model: Model, nBatches: Int = 10) where Model: GraphModel, Model.Scalar == Int32 {
        let trainBatches = dataset.normalizedFrame.batched(size: batchSize)
        for batch in dataset.normalizedNegativeFrame.batched(size: batchSize, nBatches: nBatches) {
            let negativeScores = model(batch.tensor)
            let scores = model(trainBatches.randomElement()!.tensor)
            assert(scores.sum().scalarized() < negativeScores.sum().scalarized())
        }
    }
}