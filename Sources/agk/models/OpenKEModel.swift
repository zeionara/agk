import Foundation
import TensorFlow
import PythonKit

public struct OpenKEModel<ConfigurationType>: GenericModel where ConfigurationType: PythonConvertible {
    public let configuration: ConfigurationType
    public let device: Device

    public init(configuration: ConfigurationType, device: Device = Device.default) {
        self.configuration = configuration
        self.device = device
    }

    public func callAsFunction(_ triples: Tensor<Int32>) -> Tensor<Float> {
        let data = triples.unstacked().map {
            $0.unstacked().map {
                Int($0.scalarized())
            }
        }
        var predictions = configuration.pythonObject.predict_triples(data).map {
            Float($0)!
        }

        return Tensor<Float>(predictions, on: device).reshaped(to: [-1, 1])
    }
}
