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
            dataset: KnowledgeGraphDataset<SourceElement, Int32>, metrics: [LinearMetric], train: @escaping (_ trainFrame: TripleFrame<Int32>, _ trainer: TrainerType) -> Model
    ) {
        var scores: [String: [Float]] = metrics.toDict { (metric: LinearMetric) -> (key: String, value: [Float]) in
            (metric.name, [Float]())
        }
        let group = DispatchGroup()
        let lock = NSLock()
        group.enter()

        for (i, (trainFrame, testFrame)) in dataset.normalizedFrame.cv(nFolds: nFolds).enumerated() {
            DispatchQueue.global(qos: .userInitiated).async{
                let training_start_timestamp = DispatchTime.now().uptimeNanoseconds
                let model = train(trainFrame, trainer)
                print("\(String(format: "%02d", i)): Trained model in \((DispatchTime.now().uptimeNanoseconds - training_start_timestamp) / 1_000_000_000) seconds")
                let evaluation_start_timestamp = DispatchTime.now().uptimeNanoseconds
                for metric in metrics {
//                    print("Computing \(metric.name)")
                    let value = metric.compute(model: model, trainFrame: trainFrame, testFrame: testFrame, dataset: dataset)
//                    print("Computed \(metric.name)")
                    lock.lock()
                    scores[metric.name]!.append(value)
                    lock.unlock()
//                    print("Added to the list of scores")
//                    print(scores)
                }
                print("\(String(format: "%02d", i)): Computed metrics in \((DispatchTime.now().uptimeNanoseconds - evaluation_start_timestamp) / 1_000_000_000) seconds")
//                print(scores)
                group.leave()
            }
        }

        group.wait()
        for metric in metrics {
            print("\(metric.name): \(metric.aggregate(scores: scores[metric.name]!))")
        }
    }
}

