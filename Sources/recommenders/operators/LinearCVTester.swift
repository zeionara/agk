import Foundation
import TensorFlow


public struct LinearCVTester<Model, OptimizerType> where OptimizerType: Optimizer, OptimizerType.Model: GraphModel, OptimizerType.Model == Model {
    public let trainer: LinearTrainer
    public let nFolds: Int

    public typealias OptimizationConfig = (model: Model, optimizer: OptimizerType)

    public init(nFolds: Int, nEpochs: Int, batchSize: Int) {
        trainer = LinearTrainer(nEpochs: nEpochs, batchSize: batchSize)
        self.nFolds = nFolds
    }

    public func test(dataset: KnowledgeGraphDataset, metric: LinearMetric,
                     loss: @differentiable (Tensor<Float>, Tensor<Float>, Float) -> Tensor<Float> = computeSumLoss, makeOptimizationConfig: () -> OptimizationConfig) {
        var scores: [Float] = []
        for (trainFrame, testFrame) in dataset.normalizedFrame.cv(nFolds: nFolds) {
            var optimizationConfig = makeOptimizationConfig()
            trainer.train(frame: trainFrame, model: &optimizationConfig.model, optimizer: &optimizationConfig.optimizer, loss: loss)
            scores.append(metric.compute(model: optimizationConfig.model, trainFrame: trainFrame, testFrame: testFrame))
        }
        print("\(metric.name): \(metric.aggregate(scores: scores))")
    }
}

