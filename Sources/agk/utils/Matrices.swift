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

var determinant_cache = [[Float]: Tensor<Float>]()
var determinant_cache_lock = NSLock()

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
            // let flattened = self.flattened().unstacked().map{$0.scalar! as! Float}
            // determinant_cache[flattened] = 0
            // determinant_cache[flattened] = 0
            let i = 0
            let zero = Tensor<Float>(0.0, on: device)
            let coefficients = Tensor<Float>(self).unstacked()[i].unstacked()
            let flattened = coefficients.map {
                $0.scalar!
            }
            // let lock = NSLock()
            // let group = DispatchGroup()
            // print("--")
            determinant_cache_lock.lock()
            if let cached = determinant_cache[flattened] {
                determinant_cache_lock.unlock()
                return cached
            }
            determinant_cache_lock.unlock()
            var value = Tensor<Float>(0.0, on: device)
            // group.enter()
            for (j, item) in coefficients.enumerated() {
                // DispatchQueue.global(qos: .userInitiated).async {
                let result = item != zero ? Tensor<Float>(Float(raise(-1, Int(i + j))), on: device) * item * (item != zero ? self.getMinor(withoutRow: i, withoutColumn: j).determinant : zero) : zero
                // lock.lock()
                value += result
                // lock.unlock()
                // group.leave()
                // }
            }
            // group.wait()
            determinant_cache_lock.lock()
            determinant_cache[flattened] = value
            determinant_cache_lock.unlock()
            return value
        }
    }

    var additional: Tensor<Float> {
        func run() -> [Tensor<Float>] {
            let start_time = DispatchTime.now().uptimeNanoseconds
            let lock = NSLock()
            let group = DispatchGroup()
//            let n_threads = 10
//            var n_items_per_thread = shape[0] / n_threads
            for i in (0...self.shape[0] - 1) {
//                for divisor in (0...n_threads) {
                // print("-\(i)")
                group.enter()
                DispatchQueue.global().async {
//                    print((divisor * n_items_per_thread)..<((divisor + 1) * n_items_per_thread))
//                    for i in (divisor * n_items_per_thread)..<((divisor + 1) * n_items_per_thread) {
//                    if i < shape[0] {
                    print(i)
                    let result = Tensor<Float>(
                            (0...shape[1] - 1).map { j -> Tensor<Float> in
                                // print(i, j)
                                // if j % 50 == 0 {
                                //     print(i, j)
                                // }
//                                print("     \(j)")
                                // let i = 1
                                // DispatchQueue.global(qos: .userInitiated).async {
                                Tensor<Float>(Float(raise(-1, Int(i + j))), on: device) * self.getMinor(withoutRow: i, withoutColumn: j).determinant
                                // }
                            }
                    )
                    lock.lock()
                    results[i] = result
                    lock.unlock()
//                    }
                    group.leave()
//                    print("\(i) left")
                }
//                print("\((divisor * n_items_per_thread)..<((divisor + 1) * n_items_per_thread))) left")

//                }
            }
            group.wait()
            print("Handled matrix in \((DispatchTime.now().uptimeNanoseconds - start_time) / 1_000_000_000) seconds")
            return results.map {
                $0!
            }
        }

        var results: [Tensor<Float>?] = (0...self.shape[0] - 1).map { element in
            Tensor<Float>?.none
        }
        return Tensor<Float>(
                run()
        )
    }

    var inverse: Tensor<Float> {
        Tensor<Float>(self).additional.transposed() / self.determinant
    }
}
