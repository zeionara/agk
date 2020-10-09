import Foundation
import TensorFlow

public struct LinearCVTester<Model, OptimizerType> where OptimizerType: Optimizer, OptimizerType.Model : LinearGraphModel, OptimizerType.Model == Model {
    public let trainer: LinearTrainer
    public let nFolds: Int

    public init(nFolds: Int, nEpochs: Int, batchSize: Int) {
        trainer = LinearTrainer(nEpochs: nEpochs, batchSize: batchSize)
        self.nFolds = nFolds
    }

    public func test(dataset: KnowledgeGraphDataset, metric: LinearMetric, loss: @differentiable (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float> = computeSumLoss,
                     makeModel: () -> Model, makeOptimizer: (OptimizerType.Model) -> OptimizerType) {
        var scores: [Float] = []
        for (trainFrame, testFrame) in dataset.normalizedFrame.cv(nFolds: nFolds) {
            var model = makeModel()
            var optimizer = makeOptimizer(model)
            trainer.train(frame: trainFrame, model: &model, optimizer: &optimizer, loss: loss)
            scores.append(metric.compute(model: model, trainFrame: trainFrame, testFrame: testFrame))
        }
        print("\(metric.name): \(metric.aggregate(scores: scores))")
    }
}

