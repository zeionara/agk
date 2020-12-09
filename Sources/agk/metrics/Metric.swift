public protocol Metric {
    var name: String { get }
    func compute<Model, SourceElement>(model: Model, trainFrame: TripleFrame<Int32>, testFrame: TripleFrame<Int32>, dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Int32
    func aggregate(scores: [Float]) -> Float
}
