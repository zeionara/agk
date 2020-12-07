import TensorFlow

extension LabelFrame where Element == Int32 {
    public var indices: Tensor<Int32> {
        Tensor(
                data.map {
                    Tensor(
                            Int32($0.first!),
                            on: device
                    )
                }
        )
    }

    public var labels: Tensor<Float> {
        Tensor(
                data.map {
                    Tensor(
                            Float($0.last!),
                            on: device
                    )
                }
        )
    }
}
