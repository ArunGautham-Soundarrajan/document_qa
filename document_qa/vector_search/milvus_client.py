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

    def __init__(self, uri: str, token: str, collection_name: str) -> None:
        self.client = MilvusClient(
            uri=uri,  # Cluster endpoint obtained from the console
            token=token,  # API key or a colon-separated cluster username and password
        )

        self.collection_name = collection_name

    def set_collection_name(self, collection_name: str):
        self.collection_name = collection_name

    def list_collection(self):
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
    ):
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
        self.client.drop_collection(collection_name=collection_name)

    def list_collection_info(
        self,
    ):
        return self.client.describe_collection(collection_name=self.collection_name)

    def insert_to_collection(self, data: List[Dict] | Dict):
        return self.client.insert(
            collection_name=self.collection_name, data=data, progress_bar=True
        )

    def delete_entity(self, doc_id: str | int | List[str | int]):
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
        return self.client.search(
            collection_name=self.collection_name,
            data=question_embedding,
            filter=f"doc_id in {str(doc_id)}" if doc_id else None,
            output_fields=["document_name", "page_number", "text"],
            limit=limit,
        )

    def close_connection(self):
        return self.client.close()
