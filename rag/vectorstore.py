import os
from typing import List, Dict, Any, Optional, Callable, Tuple
import pymongo
from db.connection import get_database, get_gridfs, get_mongodb_uri

# Definer nødvendige attributter her i stedet for å importere fra operations
db = get_database()
embeddings_collection = db["embeddings"]

class MongoDBVectorStore:
    """Custom MongoDB vector store implementation"""
    
    def __init__(self, embedding_function: Callable):
        self.embedding_function = embedding_function
    
    def add_texts(
        self, 
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add texts to the vector store
        
        Args:
            texts: List of text strings to embed and store
            metadatas: Metadata for each text
            
        Returns:
            List of IDs for the stored texts
        """
        import logging
        
        if not texts:
            logging.warning("No texts to add to vector store")
            return []
        
        # Ensure we have one metadata per text
        if metadatas is None:
            metadatas = [{} for _ in texts]
        else:
            if len(metadatas) != len(texts):
                logging.warning(f"Metadata length {len(metadatas)} doesn't match texts length {len(texts)}")
                # Pad with empty dicts if needed
                if len(metadatas) < len(texts):
                    metadatas.extend([{} for _ in range(len(texts) - len(metadatas))])
                else:
                    metadatas = metadatas[:len(texts)]
        
        # Validate texts are not empty
        valid_texts = []
        valid_metadatas = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            if not text or not isinstance(text, str):
                logging.warning(f"Skipping invalid text at index {i}: {text}")
                continue
                
            valid_texts.append(text)
            valid_metadatas.append(metadata)
        
        if not valid_texts:
            logging.warning("No valid texts after filtering")
            return []
            
        logging.info(f"Adding {len(valid_texts)} texts to vector store with metadata")
        for i, (text, metadata) in enumerate(zip(valid_texts[:3], valid_metadatas[:3])):
            logging.info(f"Sample text {i+1}: {text[:100]}...")
            logging.info(f"Sample metadata {i+1}: {metadata}")
            
        # Create embeddings for the texts
        try:
            embeddings = []
            doc_ids = []
            
            for i, (text, metadata) in enumerate(zip(valid_texts, valid_metadatas)):
                try:
                    # Generate embedding
                    embedding = self.embedding_function(text)
                    
                    # Store in MongoDB
                    if embedding:
                        try:
                            # Make sure metadata is a proper dict
                            if not isinstance(metadata, dict):
                                logging.warning(f"Converting metadata to dict: {metadata}")
                                metadata = {"source": str(metadata)}
                                
                            # Insert into collection
                            doc_id = embeddings_collection.insert_one({
                                "text": text, 
                                "embedding": embedding, 
                                "metadata": metadata
                            }).inserted_id
                            
                            doc_ids.append(str(doc_id))
                            logging.info(f"Added document {i+1}/{len(valid_texts)} with ID: {doc_id}")
                        except Exception as e:
                            logging.error(f"Error storing embedding {i}: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                    else:
                        logging.warning(f"Empty embedding generated for text {i+1}")
                except Exception as e:
                    logging.error(f"Error generating embedding for text {i+1}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    
            logging.info(f"Successfully added {len(doc_ids)}/{len(valid_texts)} texts to vector store")
            return doc_ids
        except Exception as e:
            logging.error(f"Error in add_texts: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        **kwargs: Any,
    ) -> List[Dict]:
        """
        Search for documents similar to the query string
        
        Args:
            query: Query string
            k: Number of documents to return
            
        Returns:
            List of documents with similarity scores
        """
        import logging
        
        logging.info(f"Similarity search for query: {query}")
        
        # Use our more detailed similarity_search_with_score and then format the results
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        
        # Extract just the documents and add the score to the metadata
        documents = []
        for doc, score in docs_and_scores:
            # Make a copy of the metadata and add the score
            metadata = {**doc["metadata"], "score": round(score, 3)}
            
            # Create the final document
            documents.append({
                "page_content": doc["page_content"],
                "metadata": metadata
            })
        
        logging.info(f"Returnerer {len(documents)} relevante dokumenter")
        return documents
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Dict, float]]:
        """
        Search for documents similar to the query string with similarity scores
        
        Args:
            query: Query string
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        import logging
        import re
        
        logging.info(f"Detaljert vektorsøk for spørring: {query}")
        
        try:
            # Ekstraher nøkkelord fra søkespørringen
            # Vi lager en enkel algoritme for å identifisere viktige norske medisinske termer
            query_lower = query.lower()
            
            # Medisinsk-relaterte nøkkelord på norsk (kan utvides)
            medical_keywords = [
                "als", "amyotrofisk", "lateral", "sklerose", "sykdom", "symptom", 
                "behandling", "medisin", "hjelpemiddel", "pustestøtte", "respirasjon",
                "pustemaskin", "bipap", "cpap", "peg", "sonde", "ernæring", "spisehjelp",
                "rullestol", "mobilitet", "gangvanske", "droppfot", "skinner", "strikk", 
                "fysioterapi", "ergoterapi", "nav", "hjelpemiddelsentral", "spasmer",
                "spastisitet", "svelg", "svelgvansker", "kommunikasjon", "talemaskin"
            ]
            
            # Finn nøkkelord i søkespørringen
            found_keywords = [word for word in medical_keywords if word in query_lower]
            logging.info(f"Fant nøkkelord i spørringen: {found_keywords}")
            
            # Gener embedding for spørringen
            query_embedding = self.embedding_function(query)
            logging.info(f"Generert embedding med lengde: {len(query_embedding)}")
            
            # Hybrid-søkestrategi: Kombiner nøkkelordssøk med vektorsøk
            if found_keywords:
                logging.info(f"Utfører hybrid-søk med nøkkelord: {found_keywords}")
                
                # Først, forsøk ett tekstbasert søk basert på nøkkelord for å finne svært relevante dokumenter
                keyword_search_results = self._keyword_search(found_keywords, limit=30)
                logging.info(f"Nøkkelordssøk fant {len(keyword_search_results)} dokumenter")
                
                # Deretter, kjør vektorsøk på alle dokumenter
                vector_search_results = self._vector_search(query_embedding, limit=50)
                logging.info(f"Vektorsøk fant {len(vector_search_results)} dokumenter")
                
                # Kombiner resultatene, med høyere vekt på nøkkelordsmatch
                combined_results = self._combine_search_results(
                    keyword_results=keyword_search_results,
                    vector_results=vector_search_results,
                    keyword_weight=1.5
                )
                
                # Post-process results to ensure diversity in content types
                processed_results = self._ensure_content_diversity(combined_results, k)
                
                return [(
                    {"page_content": doc["text"], "metadata": doc["metadata"]}, 
                    doc.get("score", 0)
                ) for doc in processed_results[:k]]
            else:
                # Hvis ingen nøkkelord blir funnet, bruk standard vektorsøk
                logging.info("Ingen spesifikke nøkkelord funnet, bruker standard vektorsøk")
                vector_results = self._vector_search(query_embedding, limit=k*2)
                
                # Post-process results to ensure diversity in content types
                processed_results = self._ensure_content_diversity(vector_results, k)
                
                return [(
                    {"page_content": doc["text"], "metadata": doc["metadata"]}, 
                    doc.get("score", 0)
                ) for doc in processed_results[:k]]
            
        except Exception as e:
            logging.error(f"Feil i similarity_search_with_score: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []
    
    def _keyword_search(self, keywords, limit=30):
        """Perform keyword-based search in MongoDB"""
        import logging
        from pymongo import DESCENDING
        
        try:
            # Build a query for MongoDB 
            search_conditions = []
            
            # Search in text content
            for keyword in keywords:
                # Case-insensitive search
                search_conditions.append({"text": {"$regex": f"\\b{keyword}\\b", "$options": "i"}})
            
            # Search in metadata (title, category, etc.)
            for keyword in keywords:
                search_conditions.append({"metadata.title": {"$regex": f"\\b{keyword}\\b", "$options": "i"}})
                search_conditions.append({"metadata.kategori": {"$regex": f"\\b{keyword}\\b", "$options": "i"}})
                search_conditions.append({"metadata.problem": {"$regex": f"\\b{keyword}\\b", "$options": "i"}})
                search_conditions.append({"metadata.tags": {"$regex": f"\\b{keyword}\\b", "$options": "i"}})
            
            # Combine conditions with OR
            query = {"$or": search_conditions}
            
            # Execute query 
            results = list(embeddings_collection.find(query).limit(limit))
            
            # Add a keyword match score based on how many keywords are present
            for doc in results:
                text_lower = doc.get("text", "").lower()
                metadata = doc.get("metadata", {})
                metadata_text = " ".join([
                    metadata.get("title", ""),
                    metadata.get("kategori", ""),
                    metadata.get("problem", ""),
                    " ".join(metadata.get("tags", []))
                ]).lower()
                
                # Count keyword matches in text and metadata
                keyword_score = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        keyword_score += 1
                    if keyword in metadata_text:
                        keyword_score += 0.5
                
                # Store the keyword match score
                doc["keyword_score"] = keyword_score
                
                # Set initial score based on keyword matches
                doc["score"] = keyword_score
            
            # Sort by keyword match score
            results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
            
            return results
        
        except Exception as e:
            logging.error(f"Feil i nøkkelordssøk: {e}")
            return []
    
    def _vector_search(self, query_embedding, limit=50):
        """Execute vector search using MongoDB Atlas or fallback method"""
        import logging
        
        try:
            # Try MongoDB Atlas vector search
            try:
                # Aggregation pipeline with vector search
                pipeline = [
                    {
                        "$search": {
                            "index": "embedding_vector_index",
                            "knnBeta": {
                                "vector": query_embedding,
                                "path": "embedding",
                                "k": limit
                            }
                        }
                    },
                    {
                        "$project": {
                            "text": 1,
                            "metadata": 1,
                            "embedding": 1,
                            "score": {
                                "$meta": "searchScore"
                            }
                        }
                    },
                    {
                        "$limit": limit
                    }
                ]
                
                logging.info("Forsøker å kjøre vektorsøk med MongoDB Atlas...")
                results = list(embeddings_collection.aggregate(pipeline))
                
                if results:
                    logging.info(f"Vektorsøk i MongoDB Atlas vellykket: {len(results)} resultater")
                    return results
            except Exception as e:
                logging.warning(f"MongoDB Atlas vektorsøk feilet: {e}. Faller tilbake til manuell metode.")
                
            # Fallback to manual cosine similarity calculation
            return self._fallback_vector_search(query_embedding, limit)
            
        except Exception as e:
            logging.error(f"Feil i vektorsøk: {e}")
            return []
    
    def _fallback_vector_search(self, query_embedding, limit=50):
        """Manual cosine similarity calculation when MongoDB Atlas is not available"""
        import logging
        import numpy as np
        
        logging.info("Utfører manuell cosine similarity beregning")
        
        try:
            # Limit to 1000 documents for performance
            all_docs = list(embeddings_collection.find().limit(1000))
            logging.info(f"Hentet {len(all_docs)} dokumenter for sammenligning")
            
            results = []
            
            for doc in all_docs:
                doc_embedding = doc.get("embedding")
                if not doc_embedding:
                    continue
                
                # Calculate cosine similarity
                try:
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    # Apply content type boosts
                    boost_factor = 1.0
                    metadata = doc.get("metadata", {})
                    
                    if metadata.get("innholdstype") == "personlig_erfaring":
                        boost_factor = 1.25
                    elif metadata.get("innholdstype") == "tips_og_triks":
                        boost_factor = 1.15
                    elif metadata.get("innholdstype") == "guide":
                        boost_factor = 1.1
                    
                    # Set the score
                    doc["score"] = similarity * boost_factor
                    
                    results.append(doc)
                    
                except Exception as e:
                    logging.error(f"Feil i cosine similarity beregning: {e}")
            
            # Sort by score
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logging.error(f"Feil i fallback vektorsøk: {e}")
            return []
    
    def _combine_search_results(self, keyword_results, vector_results, keyword_weight=1.5):
        """Combine and re-rank results from keyword and vector searches"""
        import logging
        
        # Create a map of document IDs to documents for easy lookup
        document_map = {}
        
        # Process keyword results
        for doc in keyword_results:
            doc_id = str(doc["_id"])
            document_map[doc_id] = doc
            # Apply keyword weight to scores
            doc["score"] = doc.get("score", 0) * keyword_weight
        
        # Process vector results
        for doc in vector_results:
            doc_id = str(doc["_id"])
            if doc_id in document_map:
                # If document already exists from keyword search, combine scores
                document_map[doc_id]["score"] += doc.get("score", 0)
            else:
                # Add new document from vector search
                document_map[doc_id] = doc
        
        # Convert back to list and sort by combined score
        combined_results = list(document_map.values())
        combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Log some stats about the results
        logging.info(f"Kombinert {len(keyword_results)} nøkkelordsresultater og {len(vector_results)} vektorresultater " +
                    f"til totalt {len(combined_results)} unike dokumenter")
        
        if combined_results:
            top_result = combined_results[0]
            logging.info(f"Toppresultat har score {top_result.get('score', 0)}, " +
                        f"innholdstype: {top_result.get('metadata', {}).get('innholdstype', 'ukjent')}, " +
                        f"text: {top_result.get('text', '')[:50]}...")
        
        return combined_results
    
    def _ensure_content_diversity(self, results, k):
        """Ensure diversity in results by including different content types"""
        import logging
        
        # Group documents by content type
        content_type_groups = {}
        for doc in results:
            content_type = doc.get("metadata", {}).get("innholdstype", "unknown")
            if content_type not in content_type_groups:
                content_type_groups[content_type] = []
            content_type_groups[content_type].append(doc)
        
        logging.info(f"Fordeling av innholdstyper: {[(t, len(g)) for t, g in content_type_groups.items()]}")
        
        # Ensure we have at least one personlig_erfaring if available
        final_results = []
        
        # First, include personal experiences if available
        if "personlig_erfaring" in content_type_groups and content_type_groups["personlig_erfaring"]:
            # Add at least one, but maximum 2 personal experiences
            for doc in content_type_groups["personlig_erfaring"][:min(2, len(content_type_groups["personlig_erfaring"]))]:
                final_results.append(doc)
                logging.info(f"Inkluderte personlig erfaring i resultatene")
        
        # Fill rest with top scoring documents, ensuring we don't duplicate
        added_ids = {doc["_id"] for doc in final_results}
        
        for doc in results:
            if len(final_results) >= k:
                break
                
            if doc["_id"] not in added_ids:
                final_results.append(doc)
                added_ids.add(doc["_id"])
        
        # Sort by score descending
        final_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return final_results
        
    def _fallback_similarity_search(self, query_embedding, k):
        """Fallback method for similarity search when MongoDB Atlas is not available"""
        import logging
        import numpy as np
        
        logging.info("Utfører manuell cosine similarity beregning")
        
        try:
            # Begrens søket til 1000 dokumenter for ytelse
            all_docs = list(embeddings_collection.find().limit(1000))
            logging.info(f"Hentet {len(all_docs)} dokumenter for sammenligning")
            
            results_with_scores = []
            
            for doc in all_docs:
                doc_embedding = doc.get("embedding")
                if not doc_embedding:
                    continue
                
                # Calculate cosine similarity
                try:
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    
                    # Apply boosts based on content type
                    boost_factor = 1.0
                    metadata = doc.get("metadata", {})
                    
                    if metadata.get("innholdstype") == "personlig_erfaring":
                        boost_factor = 1.25
                    elif metadata.get("innholdstype") == "tips_og_triks":
                        boost_factor = 1.15
                    elif metadata.get("innholdstype") == "guide":
                        boost_factor = 1.1
                    
                    final_score = similarity * boost_factor
                    
                    # Create result document
                    result_doc = {
                        "page_content": doc["text"],
                        "metadata": doc.get("metadata", {})
                    }
                    
                    results_with_scores.append((result_doc, final_score))
                    
                except Exception as e:
                    logging.error(f"Feil i cosine similarity beregning: {e}")
            
            # Sort by score and return top k results
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Ensure diversity in results
            diverse_results = []
            content_types_included = set()
            
            # Add at least one personal experience if available
            for doc, score in results_with_scores:
                content_type = doc["metadata"].get("innholdstype")
                if content_type == "personlig_erfaring" and "personlig_erfaring" not in content_types_included:
                    diverse_results.append((doc, score))
                    content_types_included.add("personlig_erfaring")
                    if len(diverse_results) >= 1:  # Ensure at least 1 personal experience
                        break
            
            # Fill with remaining results
            for doc, score in results_with_scores:
                if len(diverse_results) >= k:
                    break
                    
                # Check if we already have this document
                if any(doc["page_content"] == existing_doc["page_content"] for existing_doc, _ in diverse_results):
                    continue
                    
                diverse_results.append((doc, score))
            
            return diverse_results[:k]
        except Exception as e:
            logging.error(f"Feil i manuell vektorsøk: {e}")
            return []

    def find_personal_experiences(self, query: str, limit: int = 2) -> List[Dict]:
        """
        Søker spesifikt etter personlige erfaringer relatert til spørringen
        
        Args:
            query: Spørringen
            limit: Maksimalt antall personlige erfaringer å returnere
            
        Returns:
            Liste med dokumenter som inneholder personlige erfaringer
        """
        import logging
        try:
            # Opprett embedding for spørringen
            query_embedding = self.embedding_function([query])
            
            # Bygg spørring for å søke spesifikt etter personlige erfaringer
            pipeline = [
                # Matcher kun dokumenter av typen personlig_erfaring
                {"$match": {"metadata.innholdstype": "personlig_erfaring"}},
                
                # Legg til et felt med vektor-likhet
                {"$set": {
                    "score": {
                        "$function": {
                            "body": """
                            function(docVector, queryVector) {
                                if (!docVector || !queryVector) return 0;
                                
                                // Beregn dot product
                                let dotProduct = 0;
                                for (let i = 0; i < docVector.length; i++) {
                                    dotProduct += docVector[i] * queryVector[i];
                                }
                                
                                // Beregn magnitude av vektorene
                                let docMagnitude = 0;
                                let queryMagnitude = 0;
                                
                                for (let i = 0; i < docVector.length; i++) {
                                    docMagnitude += docVector[i] * docVector[i];
                                    queryMagnitude += queryVector[i] * queryVector[i];
                                }
                                
                                docMagnitude = Math.sqrt(docMagnitude);
                                queryMagnitude = Math.sqrt(queryMagnitude);
                                
                                // Unngå divisjon med null
                                if (docMagnitude === 0 || queryMagnitude === 0) return 0;
                                
                                // Returner cosinus-likhet
                                return dotProduct / (docMagnitude * queryMagnitude);
                            }
                            """,
                            "args": ["$vector", query_embedding],
                            "lang": "js"
                        }
                    }
                }},
                
                # Sorter etter likhetsscore
                {"$sort": {"score": -1}},
                
                # Begrens resultatet
                {"$limit": limit}
            ]
            
            # Utfør aggregeringen
            results = list(self.collection.aggregate(pipeline))
            
            # Logg resultat
            if results:
                logging.info(f"Fant {len(results)} personlige erfaringer relatert til '{query}'")
                for i, result in enumerate(results):
                    logging.info(f"  Personlig erfaring {i+1}: "
                               f"Score: {result.get('score', 0):.4f}, "
                               f"Tittel: {result.get('metadata', {}).get('title', 'ukjent')}")
            else:
                logging.info(f"Fant ingen personlige erfaringer relatert til '{query}'")
            
            return results
        
        except Exception as e:
            logging.error(f"Feil ved søk etter personlige erfaringer: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return []

    def hybrid_search(self, query: str, k: int = 6, keyword_weight: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        Utfører et hybrid søk som kombinerer vektor-likhet med nøkkelords-likhet
        
        Args:
            query: Spørring å søke med
            k: Antall resultater å returnere
            keyword_weight: Vekten å gi til nøkkelordssøk (0.0-1.0)
            
        Returns:
            Liste med (dokument, score) tupler sortert etter kombinert score
        """
        import logging
        import re
        from typing import Dict, List, Tuple, Set
        
        try:
            # Steg 1: Utfør vektorsøk
            vector_results = self.similarity_search_with_score(query, k=k*2)  # Hent flere for å ha rom for rerangering
            
            if not vector_results:
                logging.warning(f"Ingen vektorresultater funnet for '{query}'")
                return []
                
            # Steg 2: Hent nøkkelord fra spørringen
            keywords = self._extract_keywords(query)
            
            if not keywords:
                logging.info(f"Ingen viktige nøkkelord funnet i '{query}', bruker kun vektorsøk")
                return vector_results[:k]  # Returner bare topp k vektorresultater
                
            logging.info(f"Utfører hybrid søk med nøkkelord: {', '.join(keywords)}")
            
            # Steg 3: Rerangere resultatene basert på hybrid score
            hybrid_results = []
            
            for doc, vector_score in vector_results:
                # Beregn nøkkelordsscore
                doc_text = doc.get("page_content", "").lower()
                doc_metadata = doc.get("metadata", {})
                
                # Sjekk tekst og metadata for nøkkelord
                keyword_matches = 0
                for keyword in keywords:
                    # Sjekk i tekst
                    if keyword in doc_text:
                        keyword_matches += 1
                    
                    # Sjekk i metadata (tittel, tags, kategori)
                    for field in ["title", "tags", "kategori", "undertemaer"]:
                        field_value = doc_metadata.get(field, "")
                        if isinstance(field_value, str) and keyword in field_value.lower():
                            keyword_matches += 1
                        elif isinstance(field_value, list):
                            for value in field_value:
                                if isinstance(value, str) and keyword in value.lower():
                                    keyword_matches += 1
                
                # Normaliser nøkkelordsscore (0.0-1.0)
                max_possible_matches = len(keywords) * 2  # Antall nøkkelord * 2 (tekst + metadata)
                keyword_score = keyword_matches / max(max_possible_matches, 1)
                
                # Kombiner scores
                combined_score = (1 - keyword_weight) * vector_score + keyword_weight * keyword_score
                
                hybrid_results.append((doc, combined_score))
            
            # Steg 4: Sorter etter kombinert score og returner topp k
            hybrid_results.sort(key=lambda x: x[1], reverse=True)
            top_results = hybrid_results[:k]
            
            # Logg resultater
            logging.info(f"Hybrid søk returnerte {len(top_results)} resultater")
            for i, (doc, score) in enumerate(top_results[:3]):  # Logg topp 3
                metadata = doc.get("metadata", {})
                logging.info(
                    f"  Hybrid resultat {i+1}: "
                    f"Score: {score:.4f}, "
                    f"Type: {metadata.get('innholdstype', 'ukjent')}, "
                    f"Tittel: {metadata.get('title', 'ukjent')}"
                )
            
            return top_results
            
        except Exception as e:
            logging.error(f"Feil i hybrid_search: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return vector_results[:k] if vector_results else []  # Fallback til vektor-resultater
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Ekstraherer viktige nøkkelord fra spørringen
        
        Args:
            query: Spørringen
            
        Returns:
            Liste med nøkkelord
        """
        # Liste over norske stoppord (vanlige ord som ikke er viktige for søk)
        stopwords = {
            "og", "i", "jeg", "det", "at", "en", "et", "den", "til", "er", "som", "på", 
            "de", "med", "han", "av", "ikke", "der", "så", "var", "meg", "seg", "men", "ett", 
            "har", "om", "vi", "min", "mitt", "ha", "hadde", "hun", "nå", "over", "da", "ved", 
            "fra", "du", "ut", "sin", "dem", "oss", "opp", "man", "kan", "hans", "hvor", "eller", 
            "hva", "skal", "selv", "sjøl", "her", "alle", "vil", "bli", "ble", "blitt", "kunne", 
            "inn", "når", "være", "kom", "noen", "noe", "ville", "dere", "som", "deres", "kun", 
            "ja", "etter", "ned", "skulle", "denne", "for", "så", "bare", "være", "både", "enn",
            "hjelp", "få", "gi", "får", "gir"
        }
        
        # Fjern spesialtegn og konverter til små bokstaver
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Del opp i ord og fjern stoppord
        words = [word for word in clean_query.split() if word not in stopwords and len(word) > 2]
        
        # Fjern duplikater og behold rekkefølge
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        
        return unique_words

def get_vectorstore(embedding_function: Callable):
    """
    Get a MongoDB-based vector store
    
    Args:
        embedding_function: Embedding function to use
        
    Returns:
        A configured vector store instance
    """
    import logging
    
    # Ensure the vector search index exists
    try:
        logging.info("Forsøker å opprette vektorindeks i MongoDB...")
        
        # Check if we already have the index
        existing_indexes = embeddings_collection.list_indexes()
        has_vector_index = False
        
        for index in existing_indexes:
            if "embedding" in index["key"]:
                logging.info(f"Fant eksisterende indeks for embedding: {index['name']}")
                has_vector_index = True
                break
        
        if not has_vector_index:
            logging.info("Oppretter ny vektorindeks...")
            # For nyere MongoDB-versjoner kan vi bruke en mer moderne vektor-indeks
            try:
                # Forsøk først med moderne vektorindeks (MongoDB 4.4+)
                embedding_length = 1536  # Standard for OpenAI embeddings
                
                # Forsøk å opprette en moderne vektorindeks
                embeddings_collection.create_index(
                    [("embedding", "vector", {"dimensions": embedding_length, "similarity": "cosine"})],
                    name="embedding_vector_index"
                )
                logging.info("Opprettet moderne vektorindeks med cosine similarity")
            except Exception as e:
                logging.warning(f"Kunne ikke opprette moderne vektorindeks: {e}")
                
                # Fallback til eldre indekstype
                try:
                    embeddings_collection.create_index([("embedding", pymongo.GEOSPHERE)])
                    logging.info("Opprettet GEOSPHERE-indeks som fallback")
                except Exception as e2:
                    logging.error(f"Kunne heller ikke opprette GEOSPHERE-indeks: {e2}")
    except Exception as e:
        logging.error(f"Feil ved oppretting av indeks: {e}")
    
    return MongoDBVectorStore(embedding_function)

def add_texts_to_vectorstore(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    embedding_function: Callable
):
    """
    Add texts and their embeddings to the vector store
    
    Args:
        texts: List of text strings to embed and store
        metadatas: Metadata for each text
        embedding_function: Function that generates embeddings for text
    """
    vectorstore = get_vectorstore(embedding_function)
    return vectorstore.add_texts(texts, metadatas)
