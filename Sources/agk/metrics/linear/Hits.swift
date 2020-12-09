import TensorFlow

public func countMatchesWithDuplicates<Element>(matchedTriples: Set<[Element]>, validFrame: TripleFrame<Element>) -> Int {
    matchedTriples.map {
        validFrame.data.count($0)
    }.reduce(0, +)
}

public struct Hits: LinearMetric {
    let n: Int

    public init(n: Int, threshold: Float = 0.2) {
        self.n = n
    }

    public var name: String {
        "Hits@\(n)"
    }

    public func compute<Model, SourceElement>(
            model: Model,
            trainFrame: TripleFrame<Int32>, testFrame: TripleFrame<Int32>, dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Int32 {
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
