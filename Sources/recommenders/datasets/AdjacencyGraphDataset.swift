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
        let identityQuarter = Tensor<Int8>((0...nPairs - 1).map { i in
            Tensor<Int8>((0...nPairs - 1).map { j in
                Tensor<Int8>(i == j ? 1 : 0, on: device)
            })
        })
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
                            stacking: [identityQuarter, quarter],
                            alongAxis: 1
                    ),
                    Tensor(
                            stacking: [quarter.transposed(), identityQuarter],
                            alongAxis: 1
                    )
                ]
        ).reshaped(to: [nPairs * 2, -1])
    }

    public var degreeTensor: Tensor<Int32> {
        Tensor<Int32>(adjacencyTensor).sum(alongAxes: 0).diagonal().reshaped(to: adjacencyTensor.shape)
    }
}
