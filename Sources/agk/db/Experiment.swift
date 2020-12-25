import StORM
import MongoDBStORM

class Experiment: MongoDBStORM {
	var id				: String = ""
	var isCompleted		: Bool = false
	// var startTimestamp	: Date = Date()
    // var finishTimestamp	: Date = Date()
	// var progress		: Float = 0.0
	var metrics			: [String: Float] = [String: Float]()


	// The name of the database table
	override init() {
		super.init()
		_collection = EXPERIMENTS_COLLECTION_NAME
	}


	// The mapping that translates the database info back to the object
	// This is where you would do any validation or transformation as needed
	override func to(_ this: StORMRow) {
		id				= this.data["_id"] as? String			?? ""
        isCompleted		= this.data["isCompleted"] as? Bool	?? false
		// firstname		= this.data["firstname"] as? String		?? ""
		// lastname		= this.data["lastname"] as? String		?? ""
		// email			= this.data["email"] as? String			?? ""
		metrics 		= this.data["metrics"] as? [String: Float] ?? [String: Float]()
	}

	// A simple iteration.
	// Unfortunately necessary due to Swift's introspection limitations
	func rows() -> [Experiment] {
		var rows = [Experiment]()
		for i in 0..<self.results.rows.count {
			let row = Experiment()
			row.to(self.results.rows[i])
			rows.append(row)
		}
		return rows
	}
}
