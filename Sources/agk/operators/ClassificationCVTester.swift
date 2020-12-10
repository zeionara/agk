import Foundation
import TensorFlow
import Checkpoints


public struct ClassificationCVTester<Model, SourceElement> where Model: GenericModel, SourceElement: Hashable, Model.Scalar == Float {
    public let trainer: ConvolutionClassificationTrainer
    public let nFolds: Int

    public init(nFolds: Int, nEpochs: Int, batchSize: Int) {
        trainer = ConvolutionClassificationTrainer(nEpochs: nEpochs)
        self.nFolds = nFolds
    }

    private func testOneSplit(frame: TripleFrame<Int32>, trainLabels: LabelFrame<Int32>, testLabels: LabelFrame<Int32>, metrics: [ClassificationMetric], scores: inout [String: [Float]],
                              splitIndex i: Int,
                              dataset: KnowledgeGraphDataset<String, Int32>,
                              train: @escaping (_ frame: TripleFrame<Int32>, _ trainer: ConvolutionClassificationTrainer, _ labels: LabelFrame<Int32>) -> Model,
                              lockScoresArray: Optional<() -> Void> = Optional.none, unlockScoresArray: Optional<() -> Void> = Optional.none) {
        let training_start_timestamp = DispatchTime.now().uptimeNanoseconds
        let model = train(frame, trainer, trainLabels)
        print("\(String(format: "%02d", i)): Trained model in \((DispatchTime.now().uptimeNanoseconds - training_start_timestamp) / 1_000_000_000) seconds")
        let evaluation_start_timestamp = DispatchTime.now().uptimeNanoseconds
        for metric in metrics {
            // print("Computing \(metric.name)")
            let logits = model(dataset.tunedAdjecencyMatrixInverse).flattened().gathering(atIndices: testLabels.indices)
            let value = metric.compute(model: model, labels: testLabels.labels.unstacked().map{$0.scalar!}.map{Int32($0)}, logits: logits.unstacked().map{$0.scalar!}, dataset: dataset)
            // print("Computed \(metric.name)")
            lockScoresArray?()
            scores[metric.name]!.append(value)
            unlockScoresArray?()
            // print("Added to the list of scores")
            // print(scores)
        }
        // print("\(String(format: "%02d", i)): Computed metrics in \((DispatchTime.now().uptimeNanoseconds - evaluation_start_timestamp) / 1_000_000_000) seconds")
        // print(scores)
    }

    public func test(
            dataset: KnowledgeGraphDataset<String, Int32>, metrics: [ClassificationMetric], train: @escaping (_ trainFrame: TripleFrame<Int32>, _ trainer: ConvolutionClassificationTrainer, _ labels: LabelFrame<Int32>) -> Model,
            enableParallelism: Bool = true
    ) {
        var scores: [String: [Float]] = metrics.toDict { (metric: ClassificationMetric) -> (key: String, value: [Float]) in
            (metric.name, [Float]())
        }
        let group = enableParallelism ? DispatchGroup() : Optional.none
        let lock = enableParallelism ? NSLock() : Optional.none

        for (i, (trainLabels, testLabels)) in dataset.labelFrame!.cv(nFolds: nFolds).enumerated() {
            group?.enter()
            if enableParallelism {
                DispatchQueue.global(qos: .userInitiated).async {
                    testOneSplit(frame: dataset.normalizedFrame, trainLabels: trainLabels, testLabels: testLabels, metrics: metrics, scores: &scores, splitIndex: i, dataset: dataset.copy(), train: train) {
                        lock?.lock()
                    } unlockScoresArray: {
                        lock?.unlock()
                    }
                    group?.leave()
                }
            } else {
                testOneSplit(frame: dataset.normalizedFrame, trainLabels: trainLabels, testLabels: testLabels, metrics: metrics, scores: &scores, splitIndex: i, dataset: dataset.copy(), train: train)
                dataset.normalizedNegativeFrame.data.resetHistory()
                dataset.negativeFrame.data.resetHistory()
            }
        }

        group?.wait()
        for metric in metrics {
            print("\(metric.name): \(metric.aggregate(scores: scores[metric.name]!))")
        }
    }
}
