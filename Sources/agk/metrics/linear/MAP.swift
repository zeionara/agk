import TensorFlow

public struct MAP: LinearMetric {
    let n: Int

    public init(n: Int) {
        self.n = n
    }

    public var name: String {
        "MAP@\(n)"
    }

    public func compute<Model, SourceElement>(model: Model, trainFrame: TripleFrame<Int32>, testFrame: TripleFrame<Int32>,
            dataset: KnowledgeGraphDataset<SourceElement, Int32>) -> Float where Model: GenericModel, Model.Scalar == Int32 {
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
