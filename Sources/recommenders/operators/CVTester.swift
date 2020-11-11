import Foundation
import TensorFlow


public struct CVTester<Model, TrainerType, SourceElement> where Model: GenericModel, TrainerType: Trainer, SourceElement: Hashable {
    public let trainer: TrainerType
    public let nFolds: Int

    public init(nFolds: Int, nEpochs: Int, batchSize: Int) {
        trainer = TrainerType(nEpochs: nEpochs, batchSize: batchSize)
        self.nFolds = nFolds
    }

    private func testOneSplit(trainFrame: TripleFrame<Int32>, testFrame: TripleFrame<Int32>, metrics: [LinearMetric], scores: inout [String: [Float]],
                              splitIndex i: Int,
                              dataset: KnowledgeGraphDataset<SourceElement, Int32>,
                              train: @escaping (_ trainFrame: TripleFrame<Int32>, _ trainer: TrainerType) -> Model,
                              lockScoresArray: Optional<() -> Void> = Optional.none, unlockScoresArray: Optional<() -> Void> = Optional.none) {
        let training_start_timestamp = DispatchTime.now().uptimeNanoseconds
        let model = train(trainFrame, trainer)
        print("\(String(format: "%02d", i)): Trained model in \((DispatchTime.now().uptimeNanoseconds - training_start_timestamp) / 1_000_000_000) seconds")
        let evaluation_start_timestamp = DispatchTime.now().uptimeNanoseconds
        for metric in metrics {
            print("Computing \(metric.name)")
            let value = metric.compute(model: model, trainFrame: trainFrame, testFrame: testFrame, dataset: dataset)
            print("Computed \(metric.name)")
            lockScoresArray?()
            scores[metric.name]!.append(value)
            unlockScoresArray?()
            print("Added to the list of scores")
            print(scores)
        }
        print("\(String(format: "%02d", i)): Computed metrics in \((DispatchTime.now().uptimeNanoseconds - evaluation_start_timestamp) / 1_000_000_000) seconds")
        print(scores)
    }

    public func test(
            dataset: KnowledgeGraphDataset<SourceElement, Int32>, metrics: [LinearMetric], train: @escaping (_ trainFrame: TripleFrame<Int32>, _ trainer: TrainerType) -> Model,
            enableParallelism: Bool = true
    ) {
        var scores: [String: [Float]] = metrics.toDict { (metric: LinearMetric) -> (key: String, value: [Float]) in
            (metric.name, [Float]())
        }
        let group = enableParallelism ? DispatchGroup() : Optional.none
        let lock = enableParallelism ? NSLock() : Optional.none

        group?.enter()

        for (i, (trainFrame, testFrame)) in dataset.normalizedFrame.cv(nFolds: nFolds).enumerated() {
            if enableParallelism {
                DispatchQueue.global(qos: .userInitiated).async {
                    testOneSplit(trainFrame: trainFrame, testFrame: testFrame, metrics: metrics, scores: &scores, splitIndex: i, dataset: dataset, train: train) {
                        lock?.lock()
                    } unlockScoresArray: {
                        lock?.unlock()
                    }
                    group?.leave()
                }
            } else {
                testOneSplit(trainFrame: trainFrame, testFrame: testFrame, metrics: metrics, scores: &scores, splitIndex: i, dataset: dataset, train: train)
            }
        }

        group?.wait()
        for metric in metrics {
            print("\(metric.name): \(metric.aggregate(scores: scores[metric.name]!))")
        }
    }
}

