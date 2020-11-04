public struct RandomMetric: LinearMetric {
    let k: Float

    public init(k: Float = 2.4) {
        self.k = k
    }

    public var name: String {
        "Random metric (k=\(k))"
    }

    public func compute<Model, SourceElement, NormalizedElement>(
            model: Model, trainFrame: TripleFrame<NormalizedElement>, testFrame: TripleFrame<NormalizedElement>,
            dataset: KnowledgeGraphDataset<SourceElement, NormalizedElement>
    ) -> Float where Model: GraphModel {
        Float.random(in: 0..<1.0)
    }
}
