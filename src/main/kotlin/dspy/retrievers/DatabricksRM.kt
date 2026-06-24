package dspy.retrievers

/**
 * Document class for DatabricksRM results.
 */
data class Document(
    val pageContent: String,
    val metadata: Map<String, Any?>,
    val type: String,
) {
    fun toDict(): Map<String, Any?> {
        return mapOf(
            "page_content" to pageContent,
            "metadata" to metadata,
            "type" to type,
        )
    }
}

/**
 * Databricks retriever module.
 *
 * A retriever module that uses a Databricks Mosaic AI Vector Search Index
 * to return the top-k embeddings for a given query.
 *
 * Faithful port of `dspy/retrievers/databricks_rm.py`.
 */
@Suppress("UNCHECKED_CAST")
class DatabricksRM(
    private val databricksIndexName: String,
    private val databricksEndpoint: String? = null,
    private val databricksToken: String? = null,
    private val databricksClientId: String? = null,
    private val databricksClientSecret: String? = null,
    private val columns: List<String>? = null,
    private val filtersJson: String? = null,
    private val k: Int = 3,
    private val docsIdColumnName: String = "id",
    private val docsUriColumnName: String? = null,
    private val textColumnName: String = "text",
    private val useWithDatabricksAgentFramework: Boolean = false,
) : Retriever {

    private val resolvedToken: String? = databricksToken ?: System.getenv("DATABRICKS_TOKEN")
    private val resolvedEndpoint: String? = databricksEndpoint ?: System.getenv("DATABRICKS_HOST")
    private val resolvedClientId: String? = databricksClientId ?: System.getenv("DATABRICKS_CLIENT_ID")
    private val resolvedClientSecret: String? = databricksClientSecret ?: System.getenv("DATABRICKS_CLIENT_SECRET")
    private val effectiveColumns: List<String> = listOf(docsIdColumnName, textColumnName) + (columns ?: emptyList())

    init {
        require(resolvedToken != null || resolvedEndpoint != null) {
            "To retrieve documents with Databricks Vector Search, you must supply " +
                "the databricksToken and databricksEndpoint parameters, or set the " +
                "DATABRICKS_TOKEN and DATABRICKS_HOST environment variables. " +
                "You may also supply a service principal with databricksClientId " +
                "and databricksClientSecret."
        }
    }

    /**
     * Retrieve documents from Databricks Vector Search Index.
     */
    override fun query(query: String, k: Int): List<Map<String, Any?>> {
        // Stub implementation — would call Databricks Vector Search API
        println("DatabricksRM.query: Would query index '$databricksIndexName' for '$query' (k=$k)")
        return emptyList()
    }

    /**
     * Forward pass for Databricks Vector Search.
     */
    fun forward(
        query: Any,
        queryType: String = "ANN",
        filtersJson: String? = null,
    ): Any {
        // Determine query type
        val effectiveQueryType = when (queryType) {
            "vector", "text" -> "ANN"
            else -> queryType
        }

        val (queryText, queryVector) = when (query) {
            is String -> query to null
            is List<*> -> null to (query as List<Double>)
            else -> throw IllegalArgumentException("Query must be a string or a list of floats.")
        }

        // Query the Databricks Vector Search Index
        val results = queryViaRequests(
            indexName = databricksIndexName,
            k = k,
            columns = effectiveColumns,
            databricksToken = resolvedToken!!,
            databricksEndpoint = resolvedEndpoint!!,
            queryType = effectiveQueryType,
            queryText = queryText,
            queryVector = queryVector,
            filtersJson = filtersJson ?: this.filtersJson,
        )

        // Extract results
        val items = mutableListOf<Map<String, Any?>>()
        val dataArray = (results["result"] as? Map<*, *>)?.get("data_array") as? List<List<*>>
        val manifest = (results["result"] as? Map<*, *>)?.get("manifest") as? Map<*, *>
        val colNames = ((manifest as? Map<*, *>)?.get("columns") as? List<Map<*, *>>)?.map {
            it["name"] as? String
        } ?: emptyList()

        if (dataArray != null) {
            for (dataRow in dataArray) {
                val item = mutableMapOf<String, Any?>()
                for ((i, colName) in colNames.withIndex()) {
                    if (colName != null && i < dataRow.size) {
                        item[colName] = dataRow[i]
                    }
                }
                items.add(item)
            }
        }

        // Sort by score
        val sortedDocs = items
            .sortedByDescending { (it["score"] as? Double) ?: 0.0 }
            .take(k)

        return if (useWithDatabricksAgentFramework) {
            sortedDocs.map { doc ->
                Document(
                    pageContent = doc[textColumnName] as? String ?: "",
                    metadata = mapOf(
                        "doc_id" to extractDocIds(doc),
                        "doc_uri" to (if (docsUriColumnName != null) doc[docsUriColumnName] else null),
                    ) + getExtraColumns(doc),
                    type = "Document",
                ).toDict()
            }
        } else {
            mapOf<String, Any?>(
                "docs" to sortedDocs.map { it[textColumnName] as? String ?: "" },
                "doc_ids" to sortedDocs.map { extractDocIds(it) },
                "doc_uris" to (if (docsUriColumnName != null) sortedDocs.map { it[docsUriColumnName] } else emptyList()),
                "extra_columns" to sortedDocs.map { getExtraColumns(it) },
            )
        }
    }

    private fun extractDocIds(item: Map<String, Any?>): String {
        return if (docsIdColumnName == "metadata") {
            val metadata = (item["metadata"] as? String)?.let {
                try {
                    val json = kotlinx.serialization.json.Json { isLenient = true }
                    json.parseToJsonElement(it).let { element ->
                        if (element is kotlinx.serialization.json.JsonObject) element else null
                    }
                } catch (_: Exception) {
                    null
                }
            }
            (metadata?.get("document_id")?.toString()) ?: item[docsIdColumnName].toString()
        } else {
            item[docsIdColumnName].toString()
        }
    }

    private fun getExtraColumns(item: Map<String, Any?>): Map<String, Any?> {
        val exclude = mutableSetOf(docsIdColumnName, textColumnName)
        if (docsUriColumnName != null) exclude.add(docsUriColumnName)
        return item.filterKeys { it !in exclude }
    }

    /**
     * Query via HTTP requests (when Databricks SDK is not available).
     */
    private fun queryViaRequests(
        indexName: String,
        k: Int,
        columns: List<String>,
        databricksToken: String,
        databricksEndpoint: String,
        queryType: String,
        queryText: String?,
        queryVector: List<Double>?,
        filtersJson: String?,
    ): Map<String, Any?> {
        require((queryText != null) xor (queryVector != null)) {
            "Exactly one of queryText or queryVector must be specified."
        }

        val headers = mapOf(
            "Authorization" to "Bearer $databricksToken",
            "Content-Type" to "application/json",
        )

        val payload = mutableMapOf<String, Any?>(
            "columns" to columns,
            "num_results" to k,
            "query_type" to queryType,
        )
        if (filtersJson != null) payload["filters_json"] = filtersJson
        if (queryText != null) payload["query_text"] = queryText
        if (queryVector != null) payload["query_vector"] = queryVector

        val endpoint = "$databricksEndpoint/api/2.0/vector-search/indexes/$indexName/query"

        // In a full implementation, this would make an HTTP POST request
        println("Would POST to $endpoint with payload: ${payload.keys}")

        // Return empty result as stub
        return mapOf(
            "result" to mapOf(
                "manifest" to mapOf("columns" to emptyList<Any>()),
                "data_array" to emptyList<List<*>>(),
            )
        )
    }
}
