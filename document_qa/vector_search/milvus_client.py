from typing import Dict, List

from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

# Scheme for the pdf_documents collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=36),
    FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="page_number", dtype=DataType.INT32),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=768),
]

schema = CollectionSchema(fields=fields)


class Milvus:
    """Milvus client"""

    def __init__(self, uri: str, token: str, collection_name: str) -> None:
        """Initialise the milvus client

        :param str uri: Cluster endpoint
        :param str token: API key or a colon-separated cluster username and password
        :param str collection_name: Name of the collection to work with
        """
        self.client = MilvusClient(
            uri=uri,
            token=token,
        )

        self.collection_name = collection_name

    def set_collection_name(self, collection_name: str):
        """Set the default collection name to work with

        :param str collection_name: Name of the collection
        """
        self.collection_name = collection_name

    def list_collection(self):
        """Lists all the collection in the db

        :return _type_: List of all the collections
        """
        return self.client.list_collections()

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_params: dict = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        },
        dimension: int = 768,
    ) -> bool:
        """Create a new collection

        :param str collection_name: Name of the collection
        :param CollectionSchema schema: Schema of the collection
        :param _type_ index_params: Parameters to index each documents, defaults to { "metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}, }
        :param int dimension: Dimension of the vector field, defaults to 768
        :return bool: Status of the operation
        """
        try:
            self.client.create_collection_with_schema(
                collection_name=collection_name,
                dimension=dimension,
                schema=schema,
                auto_id=True,
                index_params=index_params,
            )
            self.collection_name = collection_name
            return True
        except Exception as e:
            raise e

    def drop_collection(self, collection_name: str):
        """Drop the collection

        :param str collection_name: Name of the collection to drop
        """
        self.client.drop_collection(collection_name=collection_name)

    def list_collection_info(
        self,
    ):
        """List information about the selected collection

        :return _type_: List of information about the collection
        """
        return self.client.describe_collection(collection_name=self.collection_name)

    def insert_to_collection(self, data: List[Dict] | Dict):
        """Insert data into collection

        :param List[Dict] | Dict data: Data in the structure specified in the schema
        :return _type_: List of index for the respective inserted data
        """
        return self.client.insert(
            collection_name=self.collection_name, data=data, progress_bar=True
        )

    def delete_entity(self, doc_id: str | int | List[str | int]):
        """Delete Entities from the collection

        :param str | int | List[str  |  int] doc_id: List of doc id to delete
        :return _type_: _description_
        """
        # query to get the list of primary keys to delete
        res = self.client.query(
            collection_name=self.collection_name,
            filter=f"doc_id == '{doc_id}'",
            output_fields=["id"],
        )
        id_list = [item["id"] for item in res]
        return self.client.delete(collection_name=self.collection_name, pks=id_list)

    def vector_search(
        self,
        question_embedding: List[float],
        limit: int = 5,
        doc_id: str | None = None,
    ):
        """Perform vector search and retrieve relavant entities

        :param List[float] question_embedding: Vectorized question to query against
        :param int limit: Top k results, defaults to 5
        :param str | None doc_id: List of document to perform search against, defaults to None
        :return _type_: Top k entities including document name, page number and text
        """
        return self.client.search(
            collection_name=self.collection_name,
            data=question_embedding,
            filter=f"doc_id in {str(doc_id)}" if doc_id else None,
            output_fields=["document_name", "page_number", "text"],
            limit=limit,
        )

    def close_connection(self):
        """Close the Milvus client connection

        :return _type_: _description_
        """
        return self.client.close()
