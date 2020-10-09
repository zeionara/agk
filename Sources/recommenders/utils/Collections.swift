extension Array where Element: Equatable {
    public func count(_ externalElement: Element) -> Int {
        self.filter { (internalElement: Element) in
            internalElement == externalElement
        }.count
    }
}
