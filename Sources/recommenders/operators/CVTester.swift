import Foundation
import TensorFlow


public struct CVTester<Model, OptimizerType, TrainerType, SourceElement>
        where OptimizerType: Optimizer, Model: GraphModel, OptimizerType.Model == Model, TrainerType: Trainer, SourceElement: Hashable {
    public let trainer: TrainerType
    public let nFolds: Int

    public typealias OptimizationConfig = (model: Model, optimizer: OptimizerType)

    public init(nFolds: Int, nEpochs: Int, batchSize: Int) {
        trainer = TrainerType(nEpochs: nEpochs, batchSize: batchSize)
        self.nFolds = nFolds
    }

    public func test(
            dataset: KnowledgeGraphDataset<SourceElement, Int32>, metrics: [LinearMetric], train: (_ trainFrame: TripleFrame<Int32>, _ trainer: TrainerType) -> Model
    ) {
        var scores: [String: [Float]] = metrics.toDict { (metric: LinearMetric) -> (key: String, value: [Float]) in
            (metric.name, [Float]())
        }
        for (trainFrame, testFrame) in dataset.normalizedFrame.cv(nFolds: nFolds) {
            let training_start_timestamp = DispatchTime.now().uptimeNanoseconds
            let model = train(trainFrame, trainer)
            print("Trained model in \((DispatchTime.now().uptimeNanoseconds - training_start_timestamp) / 1_000_000_000) seconds")
            let evaluation_start_timestamp = DispatchTime.now().uptimeNanoseconds
            for metric in metrics {
                scores[metric.name]!.append(metric.compute(model: model, trainFrame: trainFrame, testFrame: testFrame, dataset: dataset))
            }
            print("Computed metrics in \((DispatchTime.now().uptimeNanoseconds - evaluation_start_timestamp) / 1_000_000_000) seconds")
            print(scores)
        }
        for metric in metrics {
            print("\(metric.name): \(metric.aggregate(scores: scores[metric.name]!))")
        }
    }
}

