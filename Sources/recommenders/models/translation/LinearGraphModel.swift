import Foundation
import TensorFlow

public protocol LinearGraphModel: GraphModel {
    func normalizeEmbeddings() -> Self
}
