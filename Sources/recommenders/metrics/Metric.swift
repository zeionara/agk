public protocol Metric {
    var name: String { get }
    func compute<Model>(model: Model, trainFrame: TripleFrame, testFrame: TripleFrame, dataset: KnowledgeGraphDataset) -> Float where Model: GraphModel
    func aggregate(scores: [Float]) -> Float
}
