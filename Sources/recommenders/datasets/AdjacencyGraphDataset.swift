import Foundation
import TensorFlow

public func makePairs(entities: [Int32], relationships: [Int32]) -> [[Int32]] {
    var pairs: [[Int32]] = []
    entities.map { entity in
        relationships.map { relationship in
            pairs.append([entity, relationship])
        }
    }
    return pairs
}

extension TripleFrame {
    public var adjacencyTensor: Tensor<Int8> {
        let pairs = makePairs(entities: entities, relationships: relationships)
        let nPairs = pairs.count
        let zeroQuarter = Tensor<Int8>(zeros: [nPairs, nPairs])
        let quarter = Tensor<Int8>(
                pairs.map { outcomingPair in
                    Tensor<Int8>(
                            pairs.map { incomingPair in
                                Tensor<Int8>(
                                        outcomingPair[1] == incomingPair[1] && data.contains([outcomingPair[0], incomingPair[0], outcomingPair[1]]) ? 1 : 0,
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
}
