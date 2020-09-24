import Foundation
import TensorFlow

public struct KnowledgeGraphDataset {
    public let data: [[Int32]]

    static func readData(path: String) throws -> [[Int32]] {
        let dir = URL(fileURLWithPath: #file.replacingOccurrences(of: "Sources/recommenders/datasets/KnowledgeGraphDataset.swift", with: ""))
        let fileContents = try String(
                contentsOf: dir.appendingPathComponent("data").appendingPathComponent(path),
                encoding: .utf8
        )
        let data: [[Int32]] = fileContents.split(separator: "\n").map {
            String($0).split(separator: "\t").compactMap {
                Int32(String($0))
            }
        }
        return data
    }

    public init(path: String) {
        let data_ = try! KnowledgeGraphDataset.readData(path: path)

        data = data_
    }

    public var headsUnique: [Int32] {
        data[column: 0].unique()
    }

    public var tailsUnique: [Int32] {
        data[column: 1].unique()
    }
}