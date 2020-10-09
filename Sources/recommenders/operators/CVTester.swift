import Foundation
import TensorFlow


public struct CVTester<Model, OptimizerType, TrainerType> where OptimizerType: Optimizer, Model: GraphModel, OptimizerType.Model == Model, TrainerType: Trainer {
    public let trainer: TrainerType
    public let nFolds: Int

    public typealias OptimizationConfig = (model: Model, optimizer: OptimizerType)

    public init(nFolds: Int, nEpochs: Int, batchSize: Int) {
        trainer = TrainerType(nEpochs: nEpochs, batchSize: batchSize)
        self.nFolds = nFolds
    }

    public func test(dataset: KnowledgeGraphDataset, metric: LinearMetric, train: (_ trainFrame: TripleFrame, _ trainer: TrainerType) -> Model) {
        var scores: [Float] = []
        for (trainFrame, testFrame) in dataset.normalizedFrame.cv(nFolds: nFolds) {
            let model = train(trainFrame, trainer)
            scores.append(metric.compute(model: model, trainFrame: trainFrame, testFrame: testFrame, dataset: dataset))
        }
        print("\(metric.name): \(metric.aggregate(scores: scores))")
    }
}

