import Foundation
import TensorFlow

func raise<T: BinaryInteger>(_ base: T, _ power: T) -> T {
    func expBySq(_ y: T, _ x: T, _ n: T) -> T {
        precondition(n >= 0)
        if n == 0 {
            return y
        } else if n == 1 {
            return y * x
        } else if n.isMultiple(of: 2) {
            return expBySq(y, x * x, n / 2)
        } else { // n is odd
            return expBySq(y * x, x * x, (n - 1) / 2)
        }
    }

    return expBySq(1, base, power)
}

extension Tensor where Scalar: Numeric {
    func getMinor(withoutRow i: Int, withoutColumn j: Int) -> Tensor {
        let unstackedSelf = self.unstacked()
        let rows = Tensor(stacking: Array((i > 0 ? unstackedSelf[0...i - 1] : []) + (i < self.shape[0] - 1 ? unstackedSelf[i + 1...self.shape[0] - 1] : [])))
        let unstackedRows = rows.unstacked(alongAxis: 1)
        let cols = Tensor(stacking: Array((j > 0 ? unstackedRows[0...j - 1] : []) + (j < rows.shape[1] - 1 ? unstackedRows[j + 1...rows.shape[1] - 1] : []))).transposed()
        return cols
    }

    var determinant: Tensor<Float> {
        if self.shape == [1, 1] {
            return Tensor<Float>(self).flattened()
        } else if self.shape == [2, 2] {
            let flattened = Tensor<Float>(self).flattened()
            return flattened[0] * flattened[3] - flattened[1] * flattened[2]
        } else {
            let i = 0
            let zero = Tensor<Float>(0.0, on: device)
            let coefficients = Tensor<Float>(self).unstacked()[i].unstacked()
            return coefficients.enumerated().map { (j: Int, item: Tensor<Float>) -> Tensor<Float> in
                Tensor<Float>(Float(raise(-1, Int(i + j))), on: device) * item * (item != zero ? self.getMinor(withoutRow: i, withoutColumn: j).determinant : zero)
            }.reduce(Tensor<Float>(0.0, on: device), +)
        }
    }

    var additional: Tensor<Float> {
        Tensor<Float>(
                (0...self.shape[0] - 1).map { i in
                    Tensor<Float>(
                            (0...self.shape[1] - 1).map { j -> Tensor<Float> in
                                Tensor<Float>(Float(raise(-1, Int(i + j))), on: device) * self.getMinor(withoutRow: i, withoutColumn: j).determinant
                            }
                    )
                }
        )
    }

    var inverse: Tensor<Float> {
        Tensor<Float>(self).additional.transposed() / self.determinant
    }
}
