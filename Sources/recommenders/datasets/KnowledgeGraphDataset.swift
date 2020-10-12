import Foundation
import TensorFlow

public typealias CVSplit = (train: TripleFrame, test: TripleFrame)

public enum CorruptionDegree: Int {
    case none = 3
    case eitherHeadEitherTail = 2
    case headAndTail = 1
    case complete = 0
}

func getRandomNumbers(maxNumber: Int, listSize: Int) -> Set<Int> {
    var randomNumbers = Set<Int>()
    while randomNumbers.count < listSize && randomNumbers.count - 1 < maxNumber {
        let randomNumber = Int.random(in: 0...maxNumber)
        randomNumbers.insert(randomNumber)
    }
    return randomNumbers
}

public struct LabelFrame {
    let data: [[Int32]]
    let device: Device

    public init(data: [[Int32]], device: Device) {
        self.data = data
        self.device = device
    }

    public var indices: Tensor<Int32> {
        Tensor(
                data.map {
                    Tensor(
                            Int32($0.first!),
                            on: device
                    )
                }
        )
    }

    public var labels: Tensor<Float> {
        Tensor(
                data.map {
                    Tensor(
                            Float($0.last!),
                            on: device
                    )
                }
        )
    }
}

public struct TripleFrame {
    public let data: [[Int32]]
    let device: Device
    var entities_: [Int32]?
    var relationships_: [Int32]?

    public init(data: [[Int32]], device: Device, entities_: [Int32]? = Optional.none, relationships_: [Int32]? = Optional.none) {
        self.data = data
        self.device = device
        self.entities_ = entities_
        self.relationships_ = relationships_
    }

    public func cv(nFolds: Int) -> [CVSplit] {
        let enumeratedSplits = split(nChunks: nFolds).enumerated()
        return enumeratedSplits.map { (i: Int, testSplit: TripleFrame) in
            CVSplit(
                    train: TripleFrame(
                            data: enumeratedSplits.filter { (j: Int, trainPartialSplit: TripleFrame) in
                                j != i
                            }.map { (j: Int, trainPartialSplit: TripleFrame) in
                                trainPartialSplit.data
                            }.reduce([], +),
                            device: device,
                            entities_: entities,
                            relationships_: relationships
                    ),
                    test: testSplit
            )
        }
    }

    public func batched(sizes: [Int], shouldShuffle: Bool = true) -> [TripleFrame] {
        func addBatch() {
            batches.append(TripleFrame(data: batchSamples, device: device, entities_: entities, relationships_: relationships))
            i = 0
            batchSamples = []
        }

        var batches: [TripleFrame] = []
        var batchSamples: [[Int32]] = []
        var i = 0
        var currentSizeId = 0
        for sample in shouldShuffle ? data.shuffled() : data {
            if ((i == 0 && sizes[currentSizeId] == 0) || (i > 0 && i % sizes[currentSizeId] == 0)) {
                addBatch()
                if currentSizeId < data.count - 1 {
                    currentSizeId += 1
                    i = 0
                } else {
                    break
                }
            }
            batchSamples.append(sample)
            i += 1
        }
        if !batchSamples.isEmpty {
            addBatch()
        }
        return batches
    }

    public func getCombinations(k: Int) -> [TripleFrame] {
        self.data.getCombinations(k: k).map { combination in
            TripleFrame(data: combination, device: device, entities_: entities, relationships_: relationships)
        }
    }

    public func batched(size: Int, shouldShuffle: Bool = true) -> [TripleFrame] {
        assert(size > 0)

        func addBatch() {
            batches.append(TripleFrame(data: batchSamples, device: device, entities_: entities, relationships_: relationships))
            i = 0
            batchSamples = []
        }

        var batches: [TripleFrame] = []
        var batchSamples: [[Int32]] = []
        var i = 0
        for sample in shouldShuffle ? data.shuffled() : data {
            if (i % size == 0 && i > 0) {
                addBatch()
            }
            batchSamples.append(sample)
            i += 1
        }
        if !batchSamples.isEmpty {
            addBatch()
        }
        return batches
    }

    public func split(nChunks: Int, shouldShuffle: Bool = true) -> [TripleFrame] {
        assert(nChunks > 0)
        return batched(size: Int((Float(data.count) / Float(nChunks)).rounded(.up)), shouldShuffle: shouldShuffle)
    }

    public func split(proportions: [Float], shouldShuffle: Bool = true) -> [TripleFrame] {
        assert(proportions.reduce(0.0, +) == 1.0)
        return batched(
                sizes: proportions.map { ratio in
                    Int((ratio * Float(data.count)).rounded())
                },
                shouldShuffle: shouldShuffle
        )
    }

    public var entities: [Int32] {
        if let entities__ = entities_ {
            return entities__
        }
        return (data[column: 0] + data[column: 1]).unique()
    }

    public var relationships: [Int32] {
        if let relationships__ = relationships_ {
            return relationships__
        }
        return data[column: 2].unique()
    }

    public var tensor: Tensor<Int32> {
        Tensor(
                data.map {
                    Tensor(
                            $0.map {
                                Int32($0)
                            }, on: device
                    )
                }
        )
    }

    public func sampleNegativeFrame(negativeFrame: TripleFrame, n: Int = -1, corruptionDegree: CorruptionDegree = CorruptionDegree.eitherHeadEitherTail) -> TripleFrame {
        var negativeSamples: [[[Int32]]] = data.map { positiveSample in
            let corruptedTriples = negativeFrame.data.filter { negativeSample in
                (
                        negativeSample[2] == positiveSample[2] && (
                                (
                                        (negativeSample[0] != positiveSample[0] && negativeSample[1] != positiveSample[1]) && corruptionDegree == CorruptionDegree.headAndTail
                                ) || (
                                        (
                                                (negativeSample[0] == positiveSample[0] && negativeSample[1] != positiveSample[1]) ||
                                                        (negativeSample[0] != positiveSample[0] && negativeSample[1] == positiveSample[1])
                                        ) &&
                                                corruptionDegree == CorruptionDegree.eitherHeadEitherTail
                                )
                        )
                ) || (
                        (negativeSample[0] != positiveSample[0] && negativeSample[1] != positiveSample[1] && negativeSample[2] != positiveSample[2]) && corruptionDegree == CorruptionDegree.complete
                )
            }
            return n > 0 ? getRandomNumbers(maxNumber: corruptedTriples.count - 1, listSize: n).map {
                corruptedTriples[$0]
            } : corruptedTriples
        }
        return TripleFrame(data: negativeSamples.reduce([], +), device: device, entities_: entities, relationships_: relationships)
    }

    public var negative: TripleFrame {
        makeNegativeFrame(frame: self)
    }

    public func sample(size: Int) -> TripleFrame {
        TripleFrame(data: getRandomNumbers(maxNumber: data.count - 1, listSize: size).map {
            data[$0]
        }, device: device, entities_: entities, relationships_: relationships)
    }
}

public func makeNormalizationMappings<KeyType, ValueType>(source: [KeyType], destination: [ValueType]) -> (forward: Dictionary<KeyType, ValueType>, backward: Dictionary<ValueType, KeyType>) {
    (
            Dictionary(
                    uniqueKeysWithValues: zip(
                            source.map {
                                $0
                            }, destination
                    )
            ),
            Dictionary(
                    uniqueKeysWithValues: zip(
                            destination.map {
                                $0
                            }, source
                    )
            )
    )
}

public func makeNegativeFrame(frame: TripleFrame) -> TripleFrame {
    var negativeSamples: [[Int32]] = []
    for head in frame.entities {
        for tail in frame.entities {
            for relationship in frame.relationships {
                let sample = [head, tail, relationship]
                if !frame.data.contains(sample) {
                    negativeSamples.append(sample)
                }
            }
        }
    }
    return TripleFrame(data: negativeSamples, device: frame.device, entities_: frame.entities, relationships_: frame.relationships)
}


private func normalize(_ frame: TripleFrame, _ entityNormalizationMapping: [Int32: Int32], _ relationshipNormalizationMapping: [Int32: Int32]) -> TripleFrame {
    TripleFrame(
            data: frame.data.map {
                [
                    entityNormalizationMapping[$0[0]]!,
                    entityNormalizationMapping[$0[1]]!,
                    relationshipNormalizationMapping[$0[2]]!
                ]
            },
            device: frame.device
    )
}

public struct KnowledgeGraphDataset {
    public let frame: TripleFrame
    public let labelFrame: LabelFrame?
    public let normalizedFrame: TripleFrame
    public let negativeFrame: TripleFrame
//    public let sampledNegativeFrame: TripleFrame
    public let normalizedNegativeFrame: TripleFrame
//    public let normalizedSampledNegativeFrame: TripleFrame
    public let entityId2Index: [Int32: Int32]
    public let entityIndex2Id: [Int32: Int32]
    public let relationshipId2Index: [Int32: Int32]
    public let relationshipIndex2Id: [Int32: Int32]
    public let device: Device

    static func readData(path: String) throws -> [[Int32]] {
        let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/recommenders/datasets/KnowledgeGraphDataset.swift", with: ""))
        let fileContents = try String(
                contentsOf: dir.appendingPathComponent("data").appendingPathComponent(path),
                encoding: .utf8
        )
        let data: [[Int32]] = fileContents.split(separator: "\n").map {
            String($0).split(separator: "\t").compactMap {
                Int32(String($0))
            }
        }
        return data
    }


    public init(path: String, classes: String? = Optional.none, device: Device = Device.default) {
        let frame_ = TripleFrame(data: try! KnowledgeGraphDataset.readData(path: path), device: device)
        let negativeFrame_ = makeNegativeFrame(frame: frame_)
//        let sampledNegativeFrame_ = makeSampledNegativeFrame(frame: frame_, negativeFrame: negativeFrame_)

        let entityNormalizationMappings = makeNormalizationMappings(source: frame_.entities, destination: Array(0...frame_.entities.count - 1).map {
            Int32($0)
        })
        let relationshipNormalizationMappings = makeNormalizationMappings(source: frame_.relationships, destination: Array(0...frame_.entities.count - 1).map {
            Int32($0)
        })

        if let classes_ = classes {
            labelFrame = LabelFrame(
                    data: (try! KnowledgeGraphDataset.readData(path: classes_)).map { row in
                        [entityNormalizationMappings.forward[row.first!]!, row.last!]
                    }.sorted {
                        $0.first! < $1.first!
                    },
                    device: device
            )
        } else {
            labelFrame = Optional.none
        }

        self.device = device

        frame = frame_
        negativeFrame = negativeFrame_
//        sampledNegativeFrame = sampledNegativeFrame_
        normalizedFrame = normalize(frame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)
        normalizedNegativeFrame = normalize(negativeFrame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)
//        normalizedSampledNegativeFrame = normalize(sampledNegativeFrame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)

        entityId2Index = entityNormalizationMappings.forward
        entityIndex2Id = entityNormalizationMappings.backward
        relationshipId2Index = relationshipNormalizationMappings.forward
        relationshipIndex2Id = relationshipNormalizationMappings.backward
    }
}