import TensorFlow

extension TripleFrame where Element == Int32 {
    public var tensor: Tensor<Element> {
        Tensor(
                data.map {
                    Tensor(
                            $0.map {
                                Element($0)
                            },
                            on: device
                    )
                }
        )
    }
}
