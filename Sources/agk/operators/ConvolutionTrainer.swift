import TensorFlow

public protocol ConvolutionTrainer: Trainer {
    func train<Model, SourceElement>(dataset: KnowledgeGraphDataset<SourceElement, Int32>, model: inout Model, optimizer: Adam<Model>) where Model: GraphModel, Model.Scalar == Int32
}
