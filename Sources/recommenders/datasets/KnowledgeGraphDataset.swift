import Foundation
import TensorFlow

public typealias CVSplit<Element> = (train: TripleFrame<Element>, test: TripleFrame<Element>) where Element: Hashable

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

public struct LabelFrame<Element> {
    let data: [[Element]]
    let device: Device

    public init(data: [[Element]], device: Device) {
        self.data = data
        self.device = device
    }
}

public struct TripleFrame<Element> where Element: Hashable {
    public let data: [[Element]]
    let device: Device
    var entities_: [Element]?
    var relationships_: [Element]?

    public init(data: [[Element]], device: Device, entities_: [Element]? = Optional.none, relationships_: [Element]? = Optional.none) {
        self.data = data
        self.device = device
        self.entities_ = entities_
        self.relationships_ = relationships_
    }

    public func cv(nFolds: Int) -> [CVSplit<Element>] {
        let enumeratedSplits = split(nChunks: nFolds).enumerated()
        return enumeratedSplits.map { (i: Int, testSplit: TripleFrame) in
            CVSplit<Element>(
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
        var batchSamples: [[Element]] = []
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
        var batchSamples: [[Element]] = []
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

    public var entities: [Element] {
        if let entities__ = entities_ {
            return entities__
        }
        return (data[column: 0] + data[column: 1]).unique()
    }

    public var relationships: [Element] {
        if let relationships__ = relationships_ {
            return relationships__
        }
        return data[column: 2].unique()
    }

    public func sampleNegativeFrame(negativeFrame: TripleFrame<Element>, n: Int = -1, corruptionDegree: CorruptionDegree = CorruptionDegree.eitherHeadEitherTail) -> TripleFrame<Element> {
        var negativeSamples: [[[Element]]] = data.map { positiveSample in
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
        return TripleFrame<Element>(data: negativeSamples.reduce([], +), device: device, entities_: entities, relationships_: relationships)
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

public func makeNegativeFrame<Element>(frame: TripleFrame<Element>) -> TripleFrame<Element> {
    var negativeSamples: [[Element]] = []
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


private func normalize<SourceElement, NormalizedElement>(
        _ frame: TripleFrame<SourceElement>,
        _ entityNormalizationMapping: [SourceElement: NormalizedElement],
        _ relationshipNormalizationMapping: [SourceElement: NormalizedElement]
) -> TripleFrame<NormalizedElement> {
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

public struct KnowledgeGraphDataset<SourceElement, NormalizedElement> where SourceElement: Hashable, NormalizedElement: Hashable, NormalizedElement: Comparable {
    public let frame: TripleFrame<SourceElement>
    public let labelFrame: LabelFrame<NormalizedElement>?
    public let normalizedFrame: TripleFrame<NormalizedElement>
    public let negativeFrame: TripleFrame<SourceElement>
//    public let sampledNegativeFrame: TripleFrame
    public let normalizedNegativeFrame: TripleFrame<NormalizedElement>
//    public let normalizedSampledNegativeFrame: TripleFrame
    public let entityId2Index: [SourceElement: NormalizedElement]
    public let entityIndex2Id: [NormalizedElement: SourceElement]
    public let relationshipId2Index: [SourceElement: NormalizedElement]
    public let relationshipIndex2Id: [NormalizedElement: SourceElement]
    public let device: Device

    static func readData<Element>(path: String, stringToSourceElement: (String) -> Element) throws -> [[Element]] {
        let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/recommenders/datasets/KnowledgeGraphDataset.swift", with: ""))
        let fileContents = try String(
                contentsOf: dir.appendingPathComponent("data").appendingPathComponent(path),
                encoding: .utf8
        )
        let data: [[Element]] = fileContents.split(separator: "\n").map {
            String($0).split(separator: "\t").compactMap {
                stringToSourceElement(String($0))
            }
        }
        return data
    }

//    public static func stringToSourceElement(_ string: String) -> SourceElement? {
//        print(string)
//        return Optional.none
//    }
//
//    public static func intToNormalizedElement(_ item: Int) -> NormalizedElement? {
//        print(item)
//        return Optional.none
//    }

    public init(
            path: String, classes: String? = Optional.none, device: Device = Device.default,
            intToNormalizedElement: (Int) -> NormalizedElement, stringToNormalizedElement: (String) -> NormalizedElement, stringToSourceElement: (String) -> SourceElement,
            sourceToNormalizedElement: (SourceElement) -> NormalizedElement
    ) {
        print("Loading frame...")
        let frame_ = TripleFrame(data: try! KnowledgeGraphDataset.readData(path: path, stringToSourceElement: stringToSourceElement), device: device)
        print("Generating negative frame...")
        let negativeFrame_ = makeNegativeFrame(frame: frame_)
//        let sampledNegativeFrame_ = makeSampledNegativeFrame(frame: frame_, negativeFrame: negativeFrame_)

        print("Building entity normalization mappings...")
        let entityNormalizationMappings = makeNormalizationMappings(source: frame_.entities, destination: Array(0...frame_.entities.count - 1).map {
            intToNormalizedElement($0)
        })
        print("Setting up relationship normalization mappings...")
        let relationshipNormalizationMappings = makeNormalizationMappings(source: frame_.relationships, destination: Array(0...frame_.entities.count - 1).map {
            intToNormalizedElement($0)
        })

        if let classes_ = classes {
            labelFrame = LabelFrame(
                    data: (try! KnowledgeGraphDataset.readData(path: classes_) { s in stringToSourceElement(s) }).map { row in
                        [entityNormalizationMappings.forward[row.first!]!, sourceToNormalizedElement(row.last!)]
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