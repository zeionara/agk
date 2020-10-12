import TensorFlow

public struct MRR: LinearMetric {
    let n: Int

    public init(n: Int, threshold: Float = 0.2) {
        self.n = n
    }

    public var name: String {
        "MRR@\(n)"
    }

    public func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame, dataset: KnowledgeGraphDataset) -> Float where Model: GraphModel {
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
            var intermediateScores: [Float] = []
            for (i, triple) in sortedTriples[0..<min(sortedTriples.count, n)].enumerated() {
                if validFrame.data.contains(triple.element) {
                    intermediateScores.append(
                            Float(
                                    countMatchesWithDuplicates(
                                            matchedTriples: Set(sortedTriples[0...i].map {
                                                $0.element
                                            }).intersection(Set(validFrame.data)),
                                            validFrame: validFrame
                                    )
                            ) / Float(i + 1)
                    )
                }
            }
            finalScores.append(intermediateScores.count > 0 ? intermediateScores.mean : 0.0)
        }
        return aggregate(scores: finalScores)
    }
}
