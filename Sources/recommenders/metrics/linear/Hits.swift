import TensorFlow

public struct Hits: LinearMetric {
    let n: Int

    public init(n: Int) {
        self.n = n
    }

    public var name: String {
        "Hits@\(n)"
    }

    public func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame, dataset: KnowledgeGraphDataset) -> Float where Model: GraphModel {
        let validFrame = testFrame.sample(size: n)
        let corruptedFrame = validFrame.sampleNegativeFrame(negativeFrame: dataset.normalizedNegativeFrame, n: 2)
        let totalTensor = Tensor(stacking: validFrame.tensor.unstacked() + corruptedFrame.tensor.unstacked())
        let scores = model(totalTensor).unstacked().map{$0.scalarized()}
        let sortedTriples = (validFrame.data + corruptedFrame.data).enumerated().sorted(){ (lhs, rhs) in scores[lhs.offset] < scores[rhs.offset] }
//        print(Set(sortedTriples[0..<n]))
//        return 0.0
        return Float(Set(sortedTriples[0..<min(sortedTriples.count, n)].map{$0.element}).intersection(Set(validFrame.data)).count) / Float(validFrame.data.count)
    }
}
