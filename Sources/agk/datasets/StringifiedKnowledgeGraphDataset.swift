import Foundation
import TensorFlow

extension KnowledgeGraphDataset where SourceElement == String, NormalizedElement == Int32 {
    public init(path: String, device: Device) {
        self.init(path: path, device: device) { i in
            Int32(i)
        } stringToNormalizedElement: { s in
            Int32(s)!
        } stringToSourceElement: { s in
            s
        } sourceToNormalizedElement: { e in
            Int32(e)!
        }
    }

    public func copy() -> KnowledgeGraphDataset<SourceElement, NormalizedElement> {
        KnowledgeGraphDataset<String, Int32>(path: path, device: device)
    }
}
