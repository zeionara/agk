import Foundation
import SwiftQuantumComputing
import ComplexModule

public let HADAMARD = try! Matrix(
    [
        [Complex<Double>(1.0 / sqrt(2.0), 0), Complex<Double>(1.0 / sqrt(2.0), 0)],
        [Complex<Double>(1.0 / sqrt(2.0), 0), Complex<Double>(-1.0 / sqrt(2.0), 0)]
    ]
)
