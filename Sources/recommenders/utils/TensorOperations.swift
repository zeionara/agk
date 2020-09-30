import Foundation
import TensorFlow

public func initEmbeddings(dimensionality: Int, nItems: Int, device device_: Device) -> Embedding<Float> {
    Embedding(
            embeddings: normalizeWithL2(
                    tensor: Tensor<Float>(
                            randomUniform: [nItems, dimensionality],
                            lowerBound: Tensor(Float(-1.0) / Float(dimensionality), on: device_),
                            upperBound: Tensor(Float(1.0) / Float(dimensionality), on: device_),
                            on: device_
                    )
            )
    )
}
