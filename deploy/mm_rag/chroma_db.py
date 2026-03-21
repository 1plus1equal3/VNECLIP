import chromadb
from PIL import Image
from tqdm import tqdm
from embedder import *

class ImageVecDB:
    def __init__(self, db_path, collection_name="clip_retrieval"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

    def insert(self, model: EncoderTower, data: list[dict], batch_size=32, device='cpu'):
        """ Insert image embeddings into the ChromaDB collection."""
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            ids, images, metadatas = [], [], []
            for item in batch:
                image_path = item["image_path"]
                image_id = str(item["image_id"])
                image = Image.open(image_path).convert("RGB")
                ids.append(image_id)
                images.append(image)
                metadatas.append({
                    "image_path": image_path,
                    "caption_id": item["caption_id"],
                    "caption": item["caption"]
                })
            embeddings = image_embedding(model, images, device).tolist()
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        print(f"Inserted {len(data)} items into the collection.")

    def search(self, embeddings, top_k=5):
        """ Search for similar images in the ChromaDB collection."""
        results = self.collection.query(
            query_embeddings=embeddings.tolist(),
            n_results=top_k
        )
        return results

    def result_visualize(self, results):
        """ Parse the search results."""
        print(f"Search results:")
        for id, distance, metadata in zip(results["ids"][0], results["distances"][0], results["metadatas"][0]):
            print(f"ID: {id}, Distance: {distance}")
            print(f"Metadata: {metadata}")
            print("-" * 50)
        print("\n")