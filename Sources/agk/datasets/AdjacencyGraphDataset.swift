import Foundation
import TensorFlow

public func makePairs<Element>(entities: [Element], relationships: [Element]) -> [[Element]] {
    var pairs: [[Element]] = []
    entities.map { entity in
        relationships.map { relationship in
            pairs.append([entity, relationship])
        }
    }
    return pairs
}

extension TripleFrame {
    public var adjacencyTensor: Tensor<Int8> {
        let outcomingRelationships = Tensor<Int8>(
                entities.map { entity in
                    Tensor<Int8>(
                            relationships.map { relationship in
                                Tensor<Int8>(
                                        data.filter { triple in
                                            triple[0] == entity && triple[2] == relationship
                                        }.count > 0 ? 1 : 0,
                                        on: device
                                )
                            }
                    )
                }
        )
        let incomingRelationships = Tensor<Int8>(
                entities.map { entity in
                    Tensor<Int8>(
                            relationships.map { relationship in
                                Tensor<Int8>(
                                        data.filter { triple in
                                            triple[1] == entity && triple[2] == relationship
                                        }.count > 0 ? 1 : 0,
                                        on: device
                                )
                            }
                    )
                }
        )
        let t1 = Tensor(
                stacking: Tensor<Int8>(ones: [entities.count], on: device).diagonal().unstacked() +
                        outcomingRelationships.unstacked(alongAxis: 1) +
                        incomingRelationships.unstacked(alongAxis: 1)
        )
        let t2 = Tensor(
                stacking: Tensor(
                        stacking: outcomingRelationships.unstacked(alongAxis: 1) +
                                incomingRelationships.unstacked(alongAxis: 1)
                ).unstacked(alongAxis: 1) + Tensor(ones: [relationships.count * 2], on: device).diagonal().unstacked()
        )
        return Tensor(stacking: t1.unstacked(alongAxis: 1) + t2.unstacked(alongAxis: 1)).transposed()
    }

    public var adjacencyPairsTensor: Tensor<Int8> {
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

    public var adjacencySequence: Tensor<Float> {
        Tensor(
                data.map { triple in
                    Tensor<Float>(
                            entities.map { entity in
                                entity == triple[1] ? 1.0 : 0.0
                            }, on: device
                    )
                }
        )
    }
}

extension Tensor where Scalar: Numeric {
    public var degree: Self {
        print("1.0")
        var a = self.sum(alongAxes: 0)
        print("1.1")
        a = a.diagonal()
        print("1.2")
        a = a.reshaped(to: self.shape)
        print("1.3")
        return a
    }
}
