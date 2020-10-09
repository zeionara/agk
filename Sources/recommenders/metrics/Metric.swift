public protocol Metric {
    var name: String { get }
    func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame) -> Float where Model: GraphModel
    func aggregate(scores: [Float]) -> Float
}
