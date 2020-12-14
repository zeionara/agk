import TensorFlow

public protocol SaveableGraphModel {
    associatedtype DatasetType
    init(
        dataset: DatasetType,
        device: Device,
        activation: @escaping Dense<Float>.Activation
    ) throws
    func save() throws
}