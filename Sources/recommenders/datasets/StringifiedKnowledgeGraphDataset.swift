import Foundation

extension KnowledgeGraphDataset where SourceElement == String, NormalizedElement == Int32 {
    public static func stringToSourceElement(_ string: String) -> SourceElement? {
        string
    }

    public static func intToNormalizedElement(_ item: Int) -> NormalizedElement? {
        NormalizedElement(item)
    }
}
