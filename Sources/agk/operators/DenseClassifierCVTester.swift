import Foundation
import TensorFlow
import Checkpoints

public struct DenseClassifierCVTester<Model, SourceElement, GraphModelType> where Model: GenericModel, SourceElement: Hashable, Model.Scalar == Int32, GraphModelType: ConvolutionGraphModel {
    public let trainer: ClassificationTrainer
    public let nFolds: Int
    public typealias ModelInitizlizationClosure = (_ trainer: ClassificationTrainer, _ labels: LabelFrame<Int32>) throws -> DenseClassifier<GraphModelType>

    public init(nFolds: Int, nEpochs: Int, batchSize: Int) {
        trainer = ClassificationTrainer(nEpochs: nEpochs, batchSize: batchSize)
        self.nFolds = nFolds
    }

    private func testOneSplit(
        frame: TripleFrame<Int32>,
        trainLabels: LabelFrame<Int32>,
        testLabels: LabelFrame<Int32>,
        metrics: [ClassificationMetric], 
        scores: inout [String: [Float]],
        splitIndex i: Int,
        dataset: KnowledgeGraphDataset<String, Int32>,
        initModel: @escaping ModelInitizlizationClosure,
        getEntityIndices: (LabelFrame<Int32>) -> Tensor<Int32> = { $0.indices },
        lockScoresArray: Optional<() -> Void> = Optional.none,
        unlockScoresArray: Optional<() -> Void> = Optional.none
    ) throws {
        let training_start_timestamp = DispatchTime.now().uptimeNanoseconds
        let model = try initModel(trainer, trainLabels)
        print("\(String(format: "%02d", i)): Trained model in \((DispatchTime.now().uptimeNanoseconds - training_start_timestamp) / 1_000_000_000) seconds")
        // let evaluation_start_timestamp = DispatchTime.now().uptimeNanoseconds
        for metric in metrics {
            // print("Computing \(metric.name)")
            let logits = model(getEntityIndices(testLabels)).flattened()
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
            dataset: KnowledgeGraphDataset<String, Int32>,
            metrics: [ClassificationMetric],
            enableParallelism: Bool = true,
            getEntityIndices: @escaping (LabelFrame<Int32>) -> Tensor<Int32> = { $0.indices },
            initModel: @escaping ModelInitizlizationClosure
    ) throws {
        var scores: [String: [Float]] = metrics.toDict { (metric: ClassificationMetric) -> (key: String, value: [Float]) in
            (metric.name, [Float]())
        }
        let group = enableParallelism ? DispatchGroup() : Optional.none
        let lock = enableParallelism ? NSLock() : Optional.none

        for (i, (trainLabels, testLabels)) in dataset.labelFrame!.cv(nFolds: nFolds).enumerated() {
            if enableParallelism {
                group?.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        try testOneSplit(
                            frame: dataset.normalizedFrame,
                            trainLabels: trainLabels,
                            testLabels: testLabels,
                            metrics: metrics,
                            scores: &scores,
                            splitIndex: i,
                            dataset: dataset.copy(),
                            initModel: initModel,
                            getEntityIndices: getEntityIndices
                        ) {
                            lock?.lock()
                        } unlockScoresArray: {
                            lock?.unlock()
                        }
                        group?.leave()
                    } catch let error as NSError {
                        print("Cannot handle \(i)th fold due to exception: \(error.debugDescription)")
                    }
                }
            } else {
                try testOneSplit(
                    frame: dataset.normalizedFrame,
                    trainLabels: trainLabels,
                    testLabels: testLabels,
                    metrics: metrics,
                    scores: &scores,
                    splitIndex: i,
                    dataset: dataset.copy(),
                    initModel: initModel,
                    getEntityIndices: getEntityIndices
                )
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
