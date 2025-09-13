"""
Enhanced Database Service with Advanced Vector Operations
Supports multiple vector databases and advanced similarity search.
"""
import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import redis
import faiss
import pickle
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models.face_record import FaceRecord, FaceProcessingLog
from app.services.enhanced_face_processor import calculate_similarity
import time

logger = logging.getLogger(__name__)

class EnhancedDatabaseService:
    """Enhanced database service with advanced vector operations."""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///face_db.sqlite')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.faiss_index_path = os.getenv('FAISS_INDEX_PATH', './data/face_index.faiss')
        
        # Initialize connections
        self.engine = None
        self.Session = None
        self.redis_client = None
        self.faiss_index = None
        self.embedding_to_id = {}  # Maps FAISS index positions to face record IDs
        
        self._initialize_database()
        self._initialize_redis()
        self._initialize_faiss()
        
        logger.info("Enhanced database service initialized")
    
    def _initialize_database(self):
        """Initialize database connection."""
        try:
            self.engine = create_engine(self.database_url)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Database connected: {self.database_url}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            if 'redis://' in self.redis_url:
                self.redis_client = redis.from_url(self.redis_url)
                self.redis_client.ping()
                logger.info("Redis connected")
            else:
                logger.info("Redis disabled (no valid URL)")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, continuing without cache")
    
    def _initialize_faiss(self):
        """Initialize FAISS index for vector similarity search."""
        try:
            if os.path.exists(self.faiss_index_path):
                # Load existing index
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                
                # Load ID mapping
                mapping_path = self.faiss_index_path.replace('.faiss', '_mapping.pkl')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        self.embedding_to_id = pickle.load(f)
                
                logger.info(f"FAISS index loaded with {self.faiss_index.ntotal} vectors")
            else:
                # Create new index (will be initialized when first vector is added)
                self.faiss_index = None
                logger.info("FAISS index will be created on first use")
                
        except Exception as e:
            logger.warning(f"FAISS initialization failed: {e}")
            self.faiss_index = None
    
    def add_face_record(self, person_id: str, person_name: str, embedding: np.ndarray,
                       confidence_score: float, image_path: str, bbox: List[int],
                       landmarks: List[List[int]], encoding_model: str) -> FaceRecord:
        """Add a new face record with enhanced vector indexing."""
        session = self.Session()
        try:
            # Create face record
            face_record = FaceRecord(
                person_id=person_id,
                person_name=person_name,
                embedding=embedding.tolist(),
                confidence_score=confidence_score,
                image_path=image_path,
                bbox=bbox,
                landmarks=landmarks,
                encoding_model=encoding_model,
                created_at=datetime.utcnow()
            )
            
            session.add(face_record)
            session.commit()
            
            # Add to FAISS index
            self._add_to_faiss_index(face_record.id, embedding)
            
            # Cache in Redis
            self._cache_face_record(face_record)
            
            logger.info(f"Face record added for {person_name} (ID: {face_record.id})")
            return face_record
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add face record: {e}")
            raise
        finally:
            session.close()
    
    def _add_to_faiss_index(self, record_id: int, embedding: np.ndarray):
        """Add embedding to FAISS index."""
        try:
            if self.faiss_index is None:
                # Create new index
                dimension = len(embedding)
                self.faiss_index = faiss.IndexFlatL2(dimension)
                logger.info(f"Created FAISS index with dimension {dimension}")
            
            # Add vector to index
            embedding_2d = embedding.reshape(1, -1).astype('float32')
            self.faiss_index.add(embedding_2d)
            
            # Update ID mapping
            self.embedding_to_id[self.faiss_index.ntotal - 1] = record_id
            
            # Save index and mapping
            self._save_faiss_index()
            
        except Exception as e:
            logger.error(f"Failed to add to FAISS index: {e}")
    
    def _save_faiss_index(self):
        """Save FAISS index and ID mapping to disk."""
        try:
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            
            # Save ID mapping
            mapping_path = self.faiss_index_path.replace('.faiss', '_mapping.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.embedding_to_id, f)
                
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def find_similar_faces(self, query_embedding: np.ndarray, top_k: int = 5, 
                          threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Find similar faces using multiple search methods."""
        results = []
        
        # Method 1: FAISS search (fastest)
        faiss_results = self._faiss_search(query_embedding, top_k, threshold)
        
        # Method 2: Database search with fallback
        if not faiss_results:
            db_results = self._database_search(query_embedding, top_k, threshold)
            results.extend(db_results)
        else:
            results.extend(faiss_results)
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
    
    def _faiss_search(self, query_embedding: np.ndarray, top_k: int, 
                     threshold: float) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        try:
            # Search FAISS index
            query_2d = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.faiss_index.search(query_2d, min(top_k * 2, self.faiss_index.ntotal))
            
            results = []
            session = self.Session()
            
            try:
                for distance, idx in zip(distances[0], indices[0]):
                    if idx == -1:  # Invalid index
                        continue
                    
                    # Convert L2 distance to similarity score
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity >= threshold:
                        # Get face record ID
                        record_id = self.embedding_to_id.get(idx)
                        if record_id:
                            # Fetch full record
                            face_record = session.query(FaceRecord).filter_by(id=record_id).first()
                            if face_record:
                                results.append({
                                    'face_record': face_record,
                                    'similarity': similarity,
                                    'distance': float(distance),
                                    'search_method': 'faiss'
                                })
                
                return results
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _database_search(self, query_embedding: np.ndarray, top_k: int, 
                        threshold: float) -> List[Dict[str, Any]]:
        """Search using database with cosine similarity."""
        session = self.Session()
        try:
            # Get all face records (for small datasets)
            # For large datasets, consider using pgvector or similar
            face_records = session.query(FaceRecord).all()
            
            results = []
            for record in face_records:
                try:
                    stored_embedding = np.array(record.embedding)
                    similarity = calculate_similarity(query_embedding, stored_embedding)
                    
                    if similarity >= threshold:
                        results.append({
                            'face_record': record,
                            'similarity': similarity,
                            'distance': 1.0 - similarity,
                            'search_method': 'database'
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for record {record.id}: {e}")
                    continue
            
            return results
            
        finally:
            session.close()
    
    def identify_person(self, query_embedding: np.ndarray, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Identify a person from their face embedding."""
        # Check cache first
        cached_result = self._check_identification_cache(query_embedding)
        if cached_result:
            return cached_result
        
        # Search for similar faces
        similar_faces = self.find_similar_faces(query_embedding, top_k=1, threshold=threshold)
        
        if similar_faces:
            best_match = similar_faces[0]
            face_record = best_match['face_record']
            
            result = {
                'person_id': face_record.person_id,
                'person_name': face_record.person_name,
                'confidence': best_match['similarity'],
                'face_record_id': face_record.id,
                'search_method': best_match['search_method']
            }
            
            # Cache result
            self._cache_identification_result(query_embedding, result)
            
            return result
        
        return None
    
    def verify_person(self, query_embedding: np.ndarray, person_id: str, 
                     threshold: float = 0.7) -> Dict[str, Any]:
        """Verify if the face belongs to a specific person."""
        session = self.Session()
        try:
            # Get all face records for the person
            person_records = session.query(FaceRecord).filter_by(person_id=person_id).all()
            
            if not person_records:
                return {
                    'verified': False,
                    'confidence': 0.0,
                    'reason': 'Person not found in database'
                }
            
            best_similarity = 0.0
            best_record = None
            
            # Compare with all records for this person
            for record in person_records:
                try:
                    stored_embedding = np.array(record.embedding)
                    similarity = calculate_similarity(query_embedding, stored_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_record = record
                        
                except Exception as e:
                    logger.warning(f"Failed to verify against record {record.id}: {e}")
                    continue
            
            verified = best_similarity >= threshold
            
            return {
                'verified': verified,
                'confidence': best_similarity,
                'threshold': threshold,
                'person_id': person_id,
                'person_name': best_record.person_name if best_record else None,
                'matched_record_id': best_record.id if best_record else None
            }
            
        finally:
            session.close()
    
    def _cache_face_record(self, face_record: FaceRecord):
        """Cache face record in Redis."""
        if not self.redis_client:
            return
        
        try:
            cache_key = f"face_record:{face_record.id}"
            cache_data = {
                'id': face_record.id,
                'person_id': face_record.person_id,
                'person_name': face_record.person_name,
                'confidence_score': face_record.confidence_score,
                'encoding_model': face_record.encoding_model,
                'created_at': face_record.created_at.isoformat()
            }
            
            self.redis_client.setex(cache_key, 3600, json.dumps(cache_data))  # 1 hour TTL
            
        except Exception as e:
            logger.warning(f"Failed to cache face record: {e}")
    
    def _check_identification_cache(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """Check if identification result is cached."""
        if not self.redis_client:
            return None
        
        try:
            # Create a hash of the embedding for cache key
            embedding_hash = hash(query_embedding.tobytes())
            cache_key = f"identification:{embedding_hash}"
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
                
        except Exception as e:
            logger.warning(f"Failed to check identification cache: {e}")
        
        return None
    
    def _cache_identification_result(self, query_embedding: np.ndarray, result: Dict[str, Any]):
        """Cache identification result."""
        if not self.redis_client:
            return
        
        try:
            embedding_hash = hash(query_embedding.tobytes())
            cache_key = f"identification:{embedding_hash}"
            
            self.redis_client.setex(cache_key, 1800, json.dumps(result))  # 30 minute TTL
            
        except Exception as e:
            logger.warning(f"Failed to cache identification result: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        session = self.Session()
        try:
            # Basic counts
            total_records = session.query(FaceRecord).count()
            unique_persons = session.query(FaceRecord.person_id).distinct().count()
            processing_logs = session.query(FaceProcessingLog).count()
            
            # Model usage statistics
            model_stats = {}
            models = session.query(FaceRecord.encoding_model).distinct().all()
            for model_tuple in models:
                model = model_tuple[0]
                count = session.query(FaceRecord).filter_by(encoding_model=model).count()
                model_stats[model] = count
            
            # Vector index stats
            faiss_stats = {
                'index_size': self.faiss_index.ntotal if self.faiss_index else 0,
                'dimension': self.faiss_index.d if self.faiss_index else 0,
                'index_type': type(self.faiss_index).__name__ if self.faiss_index else 'None'
            }
            
            # Cache stats
            cache_stats = {}
            if self.redis_client:
                try:
                    info = self.redis_client.info()
                    cache_stats = {
                        'connected': True,
                        'keyspace_hits': info.get('keyspace_hits', 0),
                        'keyspace_misses': info.get('keyspace_misses', 0),
                        'used_memory': info.get('used_memory_human', '0')
                    }
                except:
                    cache_stats = {'connected': False}
            else:
                cache_stats = {'connected': False}
            
            return {
                'total_records': total_records,
                'unique_persons': unique_persons,
                'processing_logs': processing_logs,
                'model_statistics': model_stats,
                'vector_index': faiss_stats,
                'cache_statistics': cache_stats,
                'database_url': self.database_url.split('@')[-1] if '@' in self.database_url else 'local',
                'last_updated': datetime.utcnow().isoformat()
            }
            
        finally:
            session.close()
    
    def log_operation(self, operation_type: str, input_source: str, processing_time: float,
                     faces_detected: int, success: bool, metadata: Dict[str, Any] = None):
        """Log processing operation with enhanced metadata."""
        session = self.Session()
        try:
            log_entry = FaceProcessingLog(
                operation_type=operation_type,
                input_source=input_source,
                processing_time=processing_time,
                faces_detected=faces_detected,
                success=success,
                metadata=metadata or {},
                timestamp=datetime.utcnow()
            )
            
            session.add(log_entry)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log operation: {e}")
        finally:
            session.close()
    
    def cleanup_old_records(self, days_old: int = 30):
        """Clean up old processing logs and unused cache entries."""
        session = self.Session()
        try:
            # Clean up old logs
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            deleted_logs = session.query(FaceProcessingLog).filter(
                FaceProcessingLog.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            logger.info(f"Cleaned up {deleted_logs} old processing logs")
            
            # Clean up Redis cache
            if self.redis_client:
                # This is a basic cleanup - in production, consider using Redis TTL
                pass
            
        except Exception as e:
            session.rollback()
            logger.error(f"Cleanup failed: {e}")
        finally:
            session.close()
