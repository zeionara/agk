import Foundation
import TensorFlow

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
    let data: [[Int32]]
    let device: Device
    var entities_: [Int32]?
    var relationships_: [Int32]?

    public init(data: [[Int32]], device: Device, entities_: [Int32]? = Optional.none, relationships_: [Int32]? = Optional.none) {
        self.data = data
        self.device = device
        self.entities_ = entities_
        self.relationships_ = relationships_
    }

    public func batched(size: Int, shouldShuffle: Bool = true) -> [TripleFrame] {
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

    public func sampleNegativeFrame(negativeFrame: TripleFrame) -> TripleFrame {
        var negativeSamples = data.map { positiveSample in
            negativeFrame.data.filter { negativeSample in
                negativeSample[2] == positiveSample[2] && (
                        (negativeSample[0] == positiveSample[0] && negativeSample[1] != positiveSample[1]) ||
                                (negativeSample[0] != positiveSample[0] && negativeSample[1] == positiveSample[1])
                )
            }.randomElement()!
        }
        return TripleFrame(data: negativeSamples, device: device, entities_: entities, relationships_: relationships)
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