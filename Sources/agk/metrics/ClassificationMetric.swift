public protocol ClassificationMetric {
    var name: String { get }
    func compute<Model, SourceElement>(model: Model, trainLabels: LabelFrame<Int32>, testLabels: LabelFrame<Int32>, dataset: KnowledgeGraphDataset<SourceElement, Int32>
    ) -> Float where Model: GenericModel, Model.Scalar == Float
    func aggregate(scores: [Float]) -> Float
}
