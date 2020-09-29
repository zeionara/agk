import Foundation
import TensorFlow

public struct TripleFrame {
    let data: [[Int32]]
    let device: Device
    var entities_: [Int32]? = Optional.none
    var relationships_: [Int32]? = Optional.none

    public func batched(size: Int, shouldShuffle: Bool = true) -> [TripleFrame] {
        var batches: [TripleFrame] = []
        var batchSamples: [[Int32]] = []
        var i = 0
        for sample in shouldShuffle ? data.shuffled() : data {
            if (i % size == 0 && i > 0) {
                batches.append(TripleFrame(data: batchSamples, device: device, entities_: entities, relationships_: relationships))
                i = 0
                batchSamples = []
            }
            batchSamples.append(sample)
            i += 1
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
}

public func makeNormalizationMappings<KeyType, ValueType>(source: [KeyType], destination: [ValueType]) -> (forward: Dictionary<KeyType, ValueType>, backward: Dictionary<ValueType, KeyType>) where KeyType: BinaryInteger, ValueType: BinaryInteger {
    return (
            Dictionary(
                    uniqueKeysWithValues: zip(
                            source.map {
                                KeyType($0)
                            }, destination
                    )
            ),
            Dictionary(
                    uniqueKeysWithValues: zip(
                            destination.map {
                                ValueType($0)
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

public struct KnowledgeGraphDataset {
    public let frame: TripleFrame
    public let normalizedFrame: TripleFrame
    public let negativeFrame: TripleFrame
    public let normalizedNegativeFrame: TripleFrame
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

    public init(path: String, device: Device = Device.default) {
        let frame_ = TripleFrame(data: try! KnowledgeGraphDataset.readData(path: path), device: device)
        let negativeFrame_ = makeNegativeFrame(frame: frame_)

        let entityNormalizationMappings = makeNormalizationMappings(source: frame_.entities, destination: Array(0...frame_.entities.count - 1).map {
            Int32($0)
        })
        let relationshipNormalizationMappings = makeNormalizationMappings(source: frame_.relationships, destination: Array(0...frame_.entities.count - 1).map {
            Int32($0)
        })

        self.device = device
        frame = frame_
        negativeFrame = negativeFrame_
        normalizedFrame = TripleFrame(
                data: frame_.data.map {
                    [
                        entityNormalizationMappings.forward[$0[0]]!,
                        entityNormalizationMappings.forward[$0[1]]!,
                        relationshipNormalizationMappings.forward[$0[2]]!
                    ]
                },
                device: device
        )
        normalizedNegativeFrame = TripleFrame(
                data: frame_.data.map {
                    [
                        entityNormalizationMappings.forward[$0[0]]!,
                        entityNormalizationMappings.forward[$0[1]]!,
                        relationshipNormalizationMappings.forward[$0[2]]!
                    ]
                },
                device: device
        )

        entityId2Index = entityNormalizationMappings.forward
        entityIndex2Id = entityNormalizationMappings.backward
        relationshipId2Index = relationshipNormalizationMappings.forward
        relationshipIndex2Id = relationshipNormalizationMappings.backward
    }
}