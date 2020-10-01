import Foundation
import TensorFlow

public struct AdjacencyFrame {
    let data: [[Int32]]
    let device: Device

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

public func makePairs(triples: [[Int32]]) -> [[Int32]] {// (incomingPairs: [[Int32]], outcomingPairs: [[Int32]]) {
    var pairs: [[Int32]] = []
//    var outcomingPairs: [[Int32]] = []
// Generate present pairs
//    for triple in triples {
//        let incomingPair = [triple[1], triple[2]] // First element is an entity id and second is the relationship id
//        let outcomingPair = [triple[0], triple[2]]
//        if !incomingPairs.contains(incomingPair) {
//            incomingPairs.append(incomingPair)
//        }
//        if !outcomingPairs.contains(outcomingPair) {
//            outcomingPairs.append(outcomingPair)
//        }
//    }
// Generate all possible pairs
    let entities = (triples[column: 0] + triples[column: 1]).unique()
    let relationships = triples[column: 2].unique()
    entities.map { entity in
        relationships.map { relationship in
            pairs.append([entity, relationship])
            //            outcomingPairs.append([entity, relationship])
        }
    }
    return pairs
//    return (
//            incomingPairs: incomingPairs,
//            outcomingPairs: outcomingPairs
//    )
}

public func makeAdjacencyTensor(pairs: [[Int32]], triples: [[Int32]], device: Device = Device.default) -> Tensor<Int8> {
    let nPairs = pairs.count
    let zeroQuarter = Tensor<Int8>(zeros: [nPairs, nPairs])
    let quarter = Tensor<Int8>(
            pairs.map { outcomingPair in
                Tensor<Int8>(
                        pairs.map { incomingPair in
                            Tensor<Int8>(
                                    outcomingPair[1] == incomingPair[1] && triples.contains([outcomingPair[0], incomingPair[0], outcomingPair[1]]) ? 1 : 0,
                                    on: device
                            )
                        }
                )
            }
    )
    return Tensor(
            stacking: [
                Tensor(
                        stacking: [zeroQuarter, quarter],
                        alongAxis: 1
                ),
                Tensor(
                        stacking: [quarter.transposed(), zeroQuarter],
                        alongAxis: 1
                )
            ]
    ).reshaped(to: [nPairs * 2, -1])
}

public struct FlattenedKnowledgeGraphDataset {
    public let frame: Tensor<Int8>
    public let outcomingPair2Index: [[Int32]: Int32]
    public let outcomingPairIndex2Id: [Int32: [Int32]]
    public let incomingPair2Index: [[Int32]: Int32]
    public let incomingPairIndex2Id: [Int32: [Int32]]
    public let device: Device

    static func readData(path: String) throws -> [[Int32]] {
        let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/recommenders/datasets/FlattenedKnowledgeGraphDataset.swift", with: ""))
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
        let data_: [[Int32]] = try! FlattenedKnowledgeGraphDataset.readData(path: path)
        let pairs = makePairs(triples: data_)
//        print(pairs)
        frame = makeAdjacencyTensor(pairs: pairs, triples: data_, device: device)
        let outcomingPairNormalizationMappings = makeNormalizationMappings(source: pairs, destination: Array(0...pairs.count - 1).map {
            Int32($0)
        })
        let incomingPairNormalizationMappings = makeNormalizationMappings(source: pairs, destination: Array(pairs.count...pairs.count * 2 - 1).map {
            Int32($0)
        })
        print(incomingPairNormalizationMappings)
//        let negativeFrame_ = makeNegativeFrame(frame: frame_)
////        let sampledNegativeFrame_ = makeSampledNegativeFrame(frame: frame_, negativeFrame: negativeFrame_)
//
//        let entityNormalizationMappings = makeNormalizationMappings(source: frame_.entities, destination: Array(0...frame_.entities.count - 1).map {
//            Int32($0)
//        })
//        let relationshipNormalizationMappings = makeNormalizationMappings(source: frame_.relationships, destination: Array(0...frame_.entities.count - 1).map {
//            Int32($0)
//        })
//
        self.device = device
//
//        frame = frame_
//        negativeFrame = negativeFrame_
////        sampledNegativeFrame = sampledNegativeFrame_
//        normalizedFrame = normalize(frame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)
//        normalizedNegativeFrame = normalize(negativeFrame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)
////        normalizedSampledNegativeFrame = normalize(sampledNegativeFrame_, entityNormalizationMappings.forward, relationshipNormalizationMappings.forward)
//
        outcomingPair2Index = outcomingPairNormalizationMappings.forward
        outcomingPairIndex2Id = outcomingPairNormalizationMappings.backward
        incomingPair2Index = incomingPairNormalizationMappings.forward
        incomingPairIndex2Id = incomingPairNormalizationMappings.backward
    }
}