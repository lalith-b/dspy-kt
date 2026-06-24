package dspy.retrievers

/**
 * Base retriever interface.
 */
interface Retriever {
    /**
     * Retrieve relevant passages for a query.
     */
    fun query(query: String, k: Int = 3): List<Map<String, Any?>>

    suspend fun aquery(query: String, k: Int = 3): List<Map<String, Any?>> {
        return query(query, k)
    }
}
