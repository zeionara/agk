import Foundation
import TensorFlow

public struct TripleFrame {
    let data: [[Int32]]

    public var entities: [Int32] {
        (data[column: 0] + data[column: 1]).unique()
    }

    public var relationships: [Int32] {
        data[column: 2].unique()
    }

    public var tensor: Tensor<Int32> {
        Tensor(
                data.map {
                    Tensor(
                            $0.map {
                                Int32($0)
                            }
//                            on: Device.defaultXLA
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

public struct KnowledgeGraphDataset {
    public let frame: TripleFrame
    public let normalizedFrame: TripleFrame
    public let entityId2Index: [Int32: Int32]
    public let entityIndex2Id: [Int32: Int32]
    public let relationshipId2Index: [Int32: Int32]
    public let relationshipIndex2Id: [Int32: Int32]

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

    public init(path: String) {
        let frame_ = TripleFrame(data: try! KnowledgeGraphDataset.readData(path: path))

        let entityNormalizationMappings = makeNormalizationMappings(source: frame_.entities, destination: Array(0...frame_.entities.count - 1).map {
            Int32($0)
        })
        let relationshipNormalizationMappings = makeNormalizationMappings(source: frame_.relationships, destination: Array(0...frame_.entities.count - 1).map {
            Int32($0)
        })

        frame = frame_
        normalizedFrame = TripleFrame(
                data: frame_.data.map {
                    [
                        entityNormalizationMappings.forward[$0[0]]!,
                        entityNormalizationMappings.forward[$0[1]]!,
                        relationshipNormalizationMappings.forward[$0[2]]!
                    ]
                }
        )

        entityId2Index = entityNormalizationMappings.forward
        entityIndex2Id = entityNormalizationMappings.backward
        relationshipId2Index = relationshipNormalizationMappings.forward
        relationshipIndex2Id = relationshipNormalizationMappings.backward
    }
}