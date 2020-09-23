
import Foundation
import TensorFlow

extension Sequence where Element: Collection {
    subscript(column column: Element.Index) -> [Element.Iterator.Element] {
        return map { $0[column] }
    }
}

extension Sequence where Iterator.Element: Hashable {
    func unique() -> [Iterator.Element] {
        var seen: Set<Iterator.Element> = []
        return filter { seen.insert($0).inserted }
    }
}

public struct TensorPair<S1: TensorFlowScalar, S2: TensorFlowScalar>: KeyPathIterable {
    public var first: Tensor<S1>
    public var second: Tensor<S2>

    /// Creates from `first` and `second` tensors.
    public init(first: Tensor<S1>, second: Tensor<S2>) {
        self.first = first
        self.second = second
    }
}

enum simpleDatasetError: Error {
    case readingError(String)
}

public struct SimpleDataset<Entropy: RandomNumberGenerator> {
    private let trainData: [[Float]]
    public let testData: [[Float]]
    public var train: [TensorPair<Int32, Float>]
//    public let trainUsers: [Float]
    public let testUsers: [Float]
//    public let testData: [[Float]]
    public let testItems: [Float]
//    public let numUsers: Int
//    public let numItems: Int
    public let user2id: [Float: Int]
//    public let id2user: [Int: Float]
    public let item2id: [Float: Int]
//    public let id2item: [Int: Float]
    public let trainNegSampling: Tensor<Float>
//
    public typealias Samples = [TensorPair<Int32, Float>]
    public typealias Batches = Slices<Sampling<Samples, ArraySlice<Int>>>
    public typealias BatchedTensorPair = TensorPair<Int32, Float>
    public typealias Training = LazyMapSequence<
        TrainingEpochs<Samples, Entropy>,
        LazyMapSequence<Batches, BatchedTensorPair>
    >
//    public let trainMatrix: Samples
    public let training: Training

    static func readData(path: String) throws -> [[Float]]{
        let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/recommenders/datasets/SimpleDataset.swift", with: ""))
        let fileContents = try String(
                contentsOf: dir.appendingPathComponent("data").appendingPathComponent(path),
                encoding: .utf8
        )
        let data: [[Float]] = fileContents.split(separator: "\n").map {
            String($0).split(separator: "\t").compactMap { Float(String($0)) }
        }
        return data
    }

    static func appendSample(dataset: inout [TensorPair<Int32, Float>], userIndex: Int, itemIndex: Int, isNegative: Bool = false, rating: Float) {
        let userAndItem = Tensor<Int32>([Int32(userIndex), Int32(itemIndex)])
        dataset.append(TensorPair<Int32, Float>(first: userAndItem, second: Tensor<Float>([rating / 10.0])))
    }

    public init(trainBatchSize: Int = 1024, entropy: Entropy, trainPath: String, testPath: String, nNegativeSamples: Int = 3) {
        let trainData_ = try! SimpleDataset.readData(path: trainPath)
        testData = try! SimpleDataset.readData(path: trainPath)

        print("Data is read")

        let trainUsers = trainData_[column: 0].unique()
        let testUsers_ = testData[column: 0].unique()

        let trainItems = trainData_[column: 1].unique()
        let testItems_ = testData[column: 1].unique()

        let userIndices = 0...trainUsers.count - 1
        let user2id_ = Dictionary(uniqueKeysWithValues: zip(trainUsers, userIndices))
        let id2user = Dictionary(uniqueKeysWithValues: zip(userIndices, trainUsers))

        let itemIndices = 0...trainItems.count - 1
        let item2id_ = Dictionary(uniqueKeysWithValues: zip(trainItems, itemIndices))
        let id2item = Dictionary(uniqueKeysWithValues: zip(itemIndices, trainItems))

        var trainNegSampling_ = Tensor<Float>(zeros: [trainUsers.count, trainItems.count])

        var train_: [TensorPair<Int32, Float>] = []
        var rowCounter = 0
        var nRows = trainData_.count

        let maxNegSamplingAttempts = 100
        trainData_.map{ row in
//            print("Handling \(rowCounter) row out of \(nRows)")
            rowCounter += 1
            let userIndex = user2id_[row[0]]!
            let itemIndex = item2id_[row[1]]!
            let rating = row[2]

            // Set up matrix for negative sampling
            if rating > 0 {
                trainNegSampling_[userIndex][itemIndex] = Tensor(1.0)
            }

            // Set up dataset

            // Add positive samples
            SimpleDataset.appendSample(dataset: &train_, userIndex: userIndex, itemIndex: itemIndex, rating: rating)

            // Add negative samples
//            for _ in 0...nNegativeSamples - 1 {
//                var nNegSamplingAttempts = 0
//                var itemIndex = Int.random(in: itemIndices)
//                while ((trainNegSampling_[userIndex][itemIndex].scalarized() > 2.0) && (nNegSamplingAttempts < maxNegSamplingAttempts)){
//                    itemIndex = Int.random(in: itemIndices)
//                    nNegSamplingAttempts += 1
//                }
//                SimpleDataset.appendSample(dataset: &train_, userIndex: userIndex, itemIndex: itemIndex, isNegative: true)
//            }

        }

        print("Data is mapped")

        train = train_
//
//        self.testData = testData
//        self.numUsers = trainUsers.count
//        self.numItems = items.count
//        self.trainUsers = trainUsers
        testUsers = testUsers_
        testItems = testItems_
        user2id = user2id_
//        self.id2user = id2user
        item2id = item2id_
//        self.id2item = id2item
        trainNegSampling = trainNegSampling_
        trainData = trainData_
//
//        self.trainMatrix = dataset
        training = TrainingEpochs(
                samples: train,
                batchSize: trainBatchSize,
                entropy: entropy
        ).lazy.map { (batches: Batches) -> LazyMapSequence<Batches, BatchedTensorPair> in
            batches.lazy.map {
                TensorPair<Int32, Float> (
                        first: Tensor<Int32>($0.map(\.first)),
                        second: Tensor<Float>($0.map(\.second))
                )
            }
        }
    }
}

extension SimpleDataset where Entropy == SystemRandomNumberGenerator {
    public init(trainBatchSize: Int = 1024, trainPath: String, testPath: String) {
        self.init(
                trainBatchSize: trainBatchSize,
                entropy: SystemRandomNumberGenerator(),
                trainPath: trainPath,
                testPath: testPath
        )
    }
}