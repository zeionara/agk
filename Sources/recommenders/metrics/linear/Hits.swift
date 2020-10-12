import TensorFlow

public struct Hits: LinearMetric {
//    let threshold: Float
    let n: Int

    public init(n: Int, threshold: Float = 0.2) {
        self.n = n
//        self.threshold = threshold
    }

    public var name: String {
        "Hits@\(n)"
    }

    public func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame, dataset: KnowledgeGraphDataset) -> Float where Model: GraphModel {
        func countMatchesWithDuplicates(matchedTriples: Set<[Int32]>, validFrame: TripleFrame) -> Int {
            matchedTriples.map {
                validFrame.data.count($0)
            }.reduce(0, +)
        }

        var finalScores: [Float] = []
        for validFrame in testFrame.getCombinations(k: min(testFrame.data.count, n)) {
            let corruptedFrame = validFrame.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame)
            let totalTensor = Tensor(stacking: validFrame.tensor.unstacked() + corruptedFrame.tensor.unstacked())
            let scores = model(totalTensor).unstacked().map {
                $0.scalarized()
            }
            let sortedTriples = (validFrame.data + corruptedFrame.data).enumerated().sorted() { (lhs, rhs) in
                scores[lhs.offset] < scores[rhs.offset]
            }
//            let filteredTriples = sortedTriples.filter {
//                scores[$0.offset] < threshold
//            }
            finalScores.append(
                    Float(
                            countMatchesWithDuplicates(
                                    matchedTriples: Set(sortedTriples[0..<min(sortedTriples.count, n)].map {
                                        $0.element
                                    }).intersection(Set(validFrame.data)),
                                    validFrame: validFrame
                            )
                    ) / Float(n)
            )
        }
        return aggregate(scores: finalScores)
    }
}
