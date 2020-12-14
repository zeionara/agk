import Foundation
import TensorFlow
import Checkpoints
import Logging

public class GenericCVTester<Model, DataFrameType, TrainerType> where Model: GenericModel, Model.Scalar == Int32, DataFrameType: DataFrame, TrainerType: Trainer {
    public let trainer: TrainerType
    public let nFolds: Int
    
    public typealias ModelInitizlizationClosure = (_ trainer: TrainerType, _ labels: DataFrameType) throws -> Model
    public typealias MetricComputationClosure = (_ model: Model, _ metric: NamedMetric, _ trainFrame: DataFrameType, _ testFrame: DataFrameType, _ dataset: KnowledgeGraphDataset<String, Int32>) throws -> Float
    public typealias SamplesListGenerationClosure = (_ dataset: KnowledgeGraphDataset<String, Int32>) -> DataFrameType
    
    private var logger: Logger

    public init(nFolds: Int, nEpochs: Int, batchSize: Int, loggingLevel: Logger.Level = .info) {
        trainer = TrainerType(nEpochs: nEpochs, batchSize: batchSize)
        self.nFolds = nFolds
        self.logger = Logger(label: "cv-tester")
        self.logger.logLevel = loggingLevel
    }

    private func testOneSplit(
        frame: TripleFrame<Int32>,
        trainLabels: DataFrameType,
        testLabels: DataFrameType,
        metrics: [NamedMetric], 
        scores: inout [String: [Float]],
        splitIndex i: Int,
        dataset: KnowledgeGraphDataset<String, Int32>,
        initModel: @escaping ModelInitizlizationClosure,
        computeMetric: @escaping MetricComputationClosure,
        lockScoresArray: Optional<() -> Void> = Optional.none,
        unlockScoresArray: Optional<() -> Void> = Optional.none
    ) throws {
        let training_start_timestamp = DispatchTime.now().uptimeNanoseconds
        let model = try initModel(trainer, trainLabels)
        self.logger.debug("\(String(format: "%02d", i)): Trained model in \((DispatchTime.now().uptimeNanoseconds - training_start_timestamp) / 1_000_000_000) seconds")
        let evaluation_start_timestamp = DispatchTime.now().uptimeNanoseconds
        for metric in metrics {
            self.logger.trace("Computing \(metric.name)")
            let value = try computeMetric(model, metric, trainLabels, testLabels, dataset)
            self.logger.trace("Computed \(metric.name)")
            lockScoresArray?()
            scores[metric.name]!.append(value)
            unlockScoresArray?()
            self.logger.trace("Added to the list of scores")
            self.logger.trace("\(scores)")
        }
        self.logger.debug("\(String(format: "%02d", i)): Computed metrics in \((DispatchTime.now().uptimeNanoseconds - evaluation_start_timestamp) / 1_000_000_000) seconds")
        self.logger.debug("\(scores)")
    }

    public func test(
            dataset: KnowledgeGraphDataset<String, Int32>,
            metrics: [NamedMetric],
            enableParallelism: Bool = true,
            initModel: @escaping ModelInitizlizationClosure,
            computeMetric: @escaping MetricComputationClosure,
            getSamplesList: SamplesListGenerationClosure
    ) throws {
        var scores: [String: [Float]] = metrics.toDict { (metric: NamedMetric) -> (key: String, value: [Float]) in
            (metric.name, [Float]())
        }
        let group = enableParallelism ? DispatchGroup() : Optional.none
        let lock = enableParallelism ? NSLock() : Optional.none

        for (i, (trainLabels, testLabels)) in getSamplesList(dataset).cv(nFolds: nFolds).enumerated() {
            if enableParallelism {
                group?.enter()
                DispatchQueue.global(qos: .userInitiated).async { [self] in
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
                            computeMetric: computeMetric
                        ) {
                            lock?.lock()
                        } unlockScoresArray: {
                            lock?.unlock()
                        }
                        group?.leave()
                    } catch let error as NSError {
                        self.logger.error("Cannot handle \(i)th fold due to exception: \(error.debugDescription)")
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
                    computeMetric: computeMetric
                )
                dataset.normalizedNegativeFrame.data.resetHistory()
                dataset.negativeFrame.data.resetHistory()
            }
        }

        group?.wait()
        for metric in metrics {
            self.logger.info("\(metric.name): \(metric.aggregate(scores: scores[metric.name]!))")
        }
    }
}
