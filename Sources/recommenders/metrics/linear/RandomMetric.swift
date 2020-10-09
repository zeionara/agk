public struct RandomMetric: LinearMetric {
    let k: Float

    public init(k: Float = 2.4) {
        self.k = k
    }

    public var name: String {
        "Random metric (k=\(k))"
    }

    public func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame) -> Float where Model: LinearGraphModel {
        Float.random(in: 0..<1.0)
    }
}
