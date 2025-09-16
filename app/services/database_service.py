"""
Database service for face embeddings with vector similarity search.
"""
import numpy as np
import redis
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models.face_record import Base, FaceRecord, FaceProcessingLog
from app.config_file import Config
import faiss
import pickle
import os

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for face records and vector operations."""
    
    def __init__(self, database_url: str = None, redis_url: str = None):
        self.database_url = database_url or Config.DATABASE_URL
        self.redis_url = redis_url or Config.REDIS_URL
        
        # Initialize database
        self.engine = create_engine(self.database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize Redis
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        # Initialize FAISS index
        self.faiss_index = None
        self.id_to_person_map = {}
        self._init_faiss_index()
    
    def _init_faiss_index(self):
        """Initialize FAISS index for fast similarity search."""
        if not Config.USE_FAISS_INDEX:
            return
        
        try:
            embedding_size = Config.FACE_EMBEDDING_SIZE
            self.faiss_index = faiss.IndexFlatIP(embedding_size)  # Inner Product (cosine similarity)
            
            # Load existing index if available
            if os.path.exists(Config.FAISS_INDEX_PATH):
                self._load_faiss_index()
            else:
                self._build_faiss_index()
            
            logger.info(f"FAISS index initialized with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
    
    def _load_faiss_index(self):
        """Load FAISS index from disk."""
        try:
            self.faiss_index = faiss.read_index(Config.FAISS_INDEX_PATH)
            
            # Load ID mapping
            mapping_path = Config.FAISS_INDEX_PATH.replace('.faiss', '_mapping.pkl')
            if os.path.exists(mapping_path):
                with open(mapping_path, 'rb') as f:
                    self.id_to_person_map = pickle.load(f)
            
            logger.info("FAISS index loaded from disk")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS index from database records."""
        try:
            session = self.Session()
            records = session.query(FaceRecord).filter(FaceRecord.is_active == True).all()
            
            if not records:
                logger.info("No face records found, empty FAISS index created")
                session.close()
                return
            
            # Prepare vectors and mapping
            vectors = []
            self.id_to_person_map = {}
            
            for i, record in enumerate(records):
                embedding = np.array(record.embedding, dtype=np.float32)
                # Normalize for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                vectors.append(embedding)
                self.id_to_person_map[i] = record.id
            
            # Add to FAISS index
            vectors_array = np.array(vectors)
            self.faiss_index.add(vectors_array)
            
            # Save index to disk
            self._save_faiss_index()
            
            logger.info(f"FAISS index built with {len(vectors)} vectors")
            session.close()
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk."""
        try:
            os.makedirs(os.path.dirname(Config.FAISS_INDEX_PATH), exist_ok=True)
            faiss.write_index(self.faiss_index, Config.FAISS_INDEX_PATH)
            
            # Save ID mapping
            mapping_path = Config.FAISS_INDEX_PATH.replace('.faiss', '_mapping.pkl')
            with open(mapping_path, 'wb') as f:
                pickle.dump(self.id_to_person_map, f)
            
            logger.info("FAISS index saved to disk")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def add_face_record(self, app_id: str, person_id: str, person_name: str, embedding: np.ndarray,
                       confidence_score: float = 0.0, image_path: str = None,
                       bbox: Dict = None, landmarks: Dict = None,
                       encoding_model: str = "facenet", face_alias: str = None,
                       is_primary: bool = False) -> FaceRecord:
        """
        Add a new face record to the database.
        
        Args:
            app_id: Application identifier
            person_id: Unique identifier for the person within the app
            person_name: Name of the person
            embedding: Face embedding vector
            confidence_score: Detection confidence score
            image_path: Path to the source image
            bbox: Face bounding box coordinates
            landmarks: Facial landmarks
            encoding_model: Model used for encoding
            face_alias: Optional alias for this face
            is_primary: Whether this is the primary face for the person
            
        Returns:
            Created FaceRecord instance
        """
        session = self.Session()
        try:
            # If this is marked as primary, unset any existing primary for this person
            if is_primary:
                session.query(FaceRecord).filter(
                    FaceRecord.app_id == app_id,
                    FaceRecord.person_id == person_id,
                    FaceRecord.is_primary == True
                ).update({'is_primary': False})
            
            # Create new record
            record = FaceRecord(
                app_id=app_id,
                person_id=person_id,
                person_name=person_name,
                face_alias=face_alias,
                embedding=embedding.tolist(),
                confidence_score=confidence_score,
                image_path=image_path,
                encoding_model=encoding_model,
                is_primary=is_primary
            )
            
            if bbox:
                record.set_bbox(bbox)
            if landmarks:
                record.set_landmarks(landmarks)
            
            session.add(record)
            session.commit()
            
            # Update FAISS index
            if self.faiss_index is not None:
                normalized_embedding = embedding / np.linalg.norm(embedding)
                self.faiss_index.add(normalized_embedding.reshape(1, -1).astype(np.float32))
                self.id_to_person_map[self.faiss_index.ntotal - 1] = record.id
                self._save_faiss_index()
            
            # Cache in Redis
            if self.redis_client and Config.CACHE_EMBEDDINGS:
                cache_key = f"face_embedding:{record.id}"
                self.redis_client.setex(
                    cache_key,
                    Config.EMBEDDING_CACHE_TTL,
                    json.dumps(embedding.tolist())
                )
            
            logger.info(f"Added face record for person_id: {person_id}")
            return record
        
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add face record: {e}")
            raise
        finally:
            session.close()
    
    def find_similar_faces(self, query_embedding: np.ndarray, top_k: int = 5,
                          similarity_threshold: float = None) -> List[Tuple[FaceRecord, float]]:
        """
        Find similar faces using vector similarity search.
        
        Args:
            query_embedding: Query face embedding
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (FaceRecord, similarity_score)
        """
        if similarity_threshold is None:
            similarity_threshold = Config.SIMILARITY_THRESHOLD
        
        # Use FAISS if available
        if self.faiss_index is not None and self.faiss_index.ntotal > 0:
            return self._find_similar_faiss(query_embedding, top_k, similarity_threshold)
        else:
            return self._find_similar_database(query_embedding, top_k, similarity_threshold)
    
    def _find_similar_faiss(self, query_embedding: np.ndarray, top_k: int,
                           similarity_threshold: float) -> List[Tuple[FaceRecord, float]]:
        """Find similar faces using FAISS index."""
        try:
            # Normalize query embedding
            normalized_query = query_embedding / np.linalg.norm(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(
                normalized_query.reshape(1, -1).astype(np.float32), top_k
            )
            
            # Get corresponding records
            session = self.Session()
            results = []
            
            for score, idx in zip(scores[0], indices[0]):
                if score >= similarity_threshold and idx in self.id_to_person_map:
                    record_id = self.id_to_person_map[idx]
                    record = session.query(FaceRecord).filter(FaceRecord.id == record_id).first()
                    if record and record.is_active:
                        results.append((record, float(score)))
            
            session.close()
            return results
        
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self._find_similar_database(query_embedding, top_k, similarity_threshold)
    
    def _find_similar_database(self, query_embedding: np.ndarray, top_k: int,
                              similarity_threshold: float) -> List[Tuple[FaceRecord, float]]:
        """Find similar faces using database query (fallback)."""
        session = self.Session()
        try:
            records = session.query(FaceRecord).filter(FaceRecord.is_active == True).all()
            similarities = []
            
            for record in records:
                try:
                    # Convert embedding to numpy array
                    stored_embedding = np.array(record.embedding, dtype=np.float32)
                    
                    # Ensure query embedding is float32
                    query_embedding_f32 = query_embedding.astype(np.float32)
                    
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding_f32)
                    stored_norm = np.linalg.norm(stored_embedding)
                    
                    if query_norm == 0 or stored_norm == 0:
                        continue
                    
                    similarity = np.dot(query_embedding_f32, stored_embedding) / (query_norm * stored_norm)
                    
                    if similarity >= similarity_threshold:
                        similarities.append((record, float(similarity)))
                        
                except Exception as e:
                    logger.error(f"Error processing record {record.id} in search: {e}")
                    continue
            
            # Sort by similarity (descending) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        
        except Exception as e:
            logger.error(f"Database similarity search failed: {e}")
            return []
        finally:
            session.close()
    
    def identify_person(self, query_embedding: np.ndarray,
                       similarity_threshold: float = None) -> Optional[Tuple[FaceRecord, float]]:
        """
        Identify a person from a face embedding.
        
        Args:
            query_embedding: Query face embedding
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (FaceRecord, similarity_score) or None if no match
        """
        similar_faces = self.find_similar_faces(query_embedding, top_k=1, similarity_threshold=similarity_threshold)
        return similar_faces[0] if similar_faces else None
    
    def verify_person(self, person_id: str, query_embedding: np.ndarray,
                     similarity_threshold: float = None) -> Tuple[bool, float]:
        """
        Verify if a face embedding belongs to a specific person.
        
        Args:
            person_id: Person ID to verify against
            query_embedding: Query face embedding
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (is_verified, similarity_score)
        """
        if similarity_threshold is None:
            similarity_threshold = Config.SIMILARITY_THRESHOLD
        
        session = self.Session()
        try:
            # Get all records for this person
            records = session.query(FaceRecord).filter(
                FaceRecord.person_id == person_id,
                FaceRecord.is_active == True
            ).all()
            
            if not records:
                logger.warning(f"No records found for person_id: {person_id}")
                return False, 0.0
            
            logger.info(f"Found {len(records)} records for person_id: {person_id}")
            
            max_similarity = 0.0
            for record in records:
                try:
                    # Convert embedding to numpy array
                    stored_embedding = np.array(record.embedding, dtype=np.float32)
                    
                    # Ensure query embedding is float32
                    query_embedding_f32 = query_embedding.astype(np.float32)
                    
                    # Calculate cosine similarity
                    query_norm = np.linalg.norm(query_embedding_f32)
                    stored_norm = np.linalg.norm(stored_embedding)
                    
                    if query_norm == 0 or stored_norm == 0:
                        logger.warning("Zero norm detected in embeddings")
                        similarity = 0.0
                    else:
                        similarity = np.dot(query_embedding_f32, stored_embedding) / (query_norm * stored_norm)
                    
                    logger.info(f"Similarity for record {record.id}: {similarity:.4f} (threshold: {similarity_threshold})")
                    max_similarity = max(max_similarity, similarity)
                    
                except Exception as embed_error:
                    logger.error(f"Error processing embedding for record {record.id}: {embed_error}")
                    continue
            
            is_verified = max_similarity >= similarity_threshold
            logger.info(f"Verification result: {is_verified}, max_similarity: {max_similarity:.4f}")
            return is_verified, float(max_similarity)
        
        except Exception as e:
            logger.error(f"Person verification failed: {e}")
            return False, 0.0
        finally:
            session.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        session = self.Session()
        try:
            # Get basic counts
            total_records = session.query(FaceRecord).count()
            active_records = session.query(FaceRecord).filter(FaceRecord.is_active == True).count()
            unique_persons = session.query(FaceRecord.person_id).distinct().count()
            
            # Get recent activity (handle case where table might not exist or be empty)
            recent_logs = []
            try:
                recent_logs_query = session.query(FaceProcessingLog).order_by(
                    FaceProcessingLog.created_at.desc()
                ).limit(10).all()
                recent_logs = [log.to_dict() for log in recent_logs_query]
            except Exception as log_error:
                logger.warning(f"Could not fetch recent logs: {log_error}")
                recent_logs = []
            
            return {
                'total_face_records': total_records,
                'active_face_records': active_records,
                'unique_persons': unique_persons,
                'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
                'recent_operations': recent_logs
            }
        
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            # Return default stats instead of empty dict
            return {
                'total_face_records': 0,
                'active_face_records': 0,
                'unique_persons': 0,
                'faiss_index_size': 0,
                'recent_operations': [],
                'error': str(e)
            }
        finally:
            session.close()
    
    def log_operation(self, operation_type: str, input_source: str = None,
                     processing_time: float = None, faces_detected: int = 0,
                     success: bool = True, error_message: str = None,
                     metadata: Dict = None):
        """Log a face processing operation."""
        session = self.Session()
        try:
            log_entry = FaceProcessingLog(
                operation_type=operation_type,
                input_source=input_source,
                processing_time=processing_time,
                faces_detected=faces_detected,
                success=success,
                error_message=error_message
            )
            
            if metadata:
                log_entry.set_metadata(metadata)
            
            session.add(log_entry)
            session.commit()
        
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to log operation: {e}")
        finally:
            session.close()
    
    def delete_face_record(self, record_id: int) -> bool:
        """
        Delete a face record (soft delete).
        
        Args:
            record_id: ID of the record to delete
            
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            record = session.query(FaceRecord).filter(FaceRecord.id == record_id).first()
            if record:
                record.is_active = False
                session.commit()
                
                # Remove from cache
                if self.redis_client:
                    cache_key = f"face_embedding:{record_id}"
                    self.redis_client.delete(cache_key)
                
                logger.info(f"Deactivated face record ID: {record_id}")
                return True
            return False
        
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete face record: {e}")
            return False
        finally:
            session.close()

    def get_person_faces(self, app_id: str, person_id: str) -> List[FaceRecord]:
        """
        Get all face records for a specific person in an app.
        
        Args:
            app_id: Application identifier
            person_id: Person identifier
            
        Returns:
            List of FaceRecord instances
        """
        session = self.Session()
        try:
            records = session.query(FaceRecord).filter(
                FaceRecord.app_id == app_id,
                FaceRecord.person_id == person_id,
                FaceRecord.is_active == True
            ).order_by(FaceRecord.is_primary.desc(), FaceRecord.created_at).all()
            return records
        except Exception as e:
            logger.error(f"Failed to get person faces: {e}")
            return []
        finally:
            session.close()

    def get_app_persons(self, app_id: str) -> List[Dict[str, Any]]:
        """
        Get all persons in an app with their face counts.
        
        Args:
            app_id: Application identifier
            
        Returns:
            List of person dictionaries with face counts
        """
        session = self.Session()
        try:
            # Get distinct persons with face counts
            result = session.query(
                FaceRecord.person_id,
                FaceRecord.person_name,
                session.query(FaceRecord.id).filter(
                    FaceRecord.app_id == app_id,
                    FaceRecord.person_id == FaceRecord.person_id,
                    FaceRecord.is_active == True
                ).count().label('face_count')
            ).filter(
                FaceRecord.app_id == app_id,
                FaceRecord.is_active == True
            ).group_by(FaceRecord.person_id, FaceRecord.person_name).all()
            
            persons = []
            for person_id, person_name, face_count in result:
                # Get primary face info
                primary_face = session.query(FaceRecord).filter(
                    FaceRecord.app_id == app_id,
                    FaceRecord.person_id == person_id,
                    FaceRecord.is_primary == True,
                    FaceRecord.is_active == True
                ).first()
                
                persons.append({
                    'person_id': person_id,
                    'person_name': person_name,
                    'face_count': face_count,
                    'primary_face_id': primary_face.id if primary_face else None,
                    'has_primary': primary_face is not None
                })
            
            return persons
        except Exception as e:
            logger.error(f"Failed to get app persons: {e}")
            return []
        finally:
            session.close()

    def set_primary_face(self, app_id: str, person_id: str, face_record_id: int) -> bool:
        """
        Set a face as the primary face for a person.
        
        Args:
            app_id: Application identifier
            person_id: Person identifier
            face_record_id: Face record ID to set as primary
            
        Returns:
            True if successful, False otherwise
        """
        session = self.Session()
        try:
            # First, unset any existing primary face
            session.query(FaceRecord).filter(
                FaceRecord.app_id == app_id,
                FaceRecord.person_id == person_id,
                FaceRecord.is_primary == True
            ).update({'is_primary': False})
            
            # Set the new primary face
            result = session.query(FaceRecord).filter(
                FaceRecord.app_id == app_id,
                FaceRecord.person_id == person_id,
                FaceRecord.id == face_record_id,
                FaceRecord.is_active == True
            ).update({'is_primary': True})
            
            session.commit()
            return result > 0
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to set primary face: {e}")
            return False
        finally:
            session.close()

    def find_best_match_per_person(self, app_id: str, query_embedding: np.ndarray, 
                                  similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find the best matching face for each person in the app.
        
        Args:
            app_id: Application identifier
            query_embedding: Query face embedding
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of best matches per person
        """
        session = self.Session()
        try:
            # Get all active faces for the app
            faces = session.query(FaceRecord).filter(
                FaceRecord.app_id == app_id,
                FaceRecord.is_active == True
            ).all()
            
            if not faces:
                return []
            
            # Calculate similarities and group by person
            person_matches = {}
            for face in faces:
                try:
                    face_embedding = np.array(face.embedding, dtype=np.float32)
                    similarity = self._calculate_cosine_similarity(query_embedding, face_embedding)
                    
                    if similarity >= similarity_threshold:
                        if face.person_id not in person_matches or similarity > person_matches[face.person_id]['similarity']:
                            person_matches[face.person_id] = {
                                'face_record_id': face.id,
                                'person_id': face.person_id,
                                'person_name': face.person_name,
                                'face_alias': face.face_alias,
                                'similarity': similarity,
                                'confidence_score': face.confidence_score,
                                'is_primary': face.is_primary
                            }
                except Exception as e:
                    logger.warning(f"Error calculating similarity for face {face.id}: {e}")
                    continue
            
            # Return sorted by similarity
            return sorted(person_matches.values(), key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to find best matches: {e}")
            return []
        finally:
            session.close()

    def validate_face_ownership(self, app_id: str, person_id: str, new_embedding: np.ndarray,
                               similarity_threshold: float = 0.5) -> bool:
        """
        Validate that a new face actually belongs to the specified person.
        
        Enhanced validation with multiple strategies:
        1. Average similarity check with lower threshold for same person variations
        2. Best match check (highest similarity)
        3. Statistical validation using multiple existing faces
        
        Args:
            app_id: Application identifier
            person_id: Person identifier
            new_embedding: New face embedding to validate
            similarity_threshold: Minimum similarity to existing faces (lowered from 0.7 to 0.5)
            
        Returns:
            True if the face likely belongs to the person, False otherwise
        """
        session = self.Session()
        try:
            existing_faces = session.query(FaceRecord).filter(
                FaceRecord.app_id == app_id,
                FaceRecord.person_id == person_id,
                FaceRecord.is_active == True
            ).all()
            
            if not existing_faces:
                # No existing faces, so we can't validate
                return True
            
            similarities = []
            
            # Check similarity against existing faces
            for face in existing_faces:
                try:
                    face_embedding = np.array(face.embedding, dtype=np.float32)
                    similarity = self._calculate_cosine_similarity(new_embedding, face_embedding)
                    similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Error validating face ownership for face {face.id}: {e}")
                    continue
            
            if not similarities:
                return False
            
            # Enhanced validation strategies
            max_similarity = max(similarities)
            avg_similarity = sum(similarities) / len(similarities)
            
            # Strategy 1: If best match is above threshold, accept
            if max_similarity >= similarity_threshold:
                logger.info(f"Face validation passed - best match: {max_similarity:.3f}")
                return True
            
            # Strategy 2: If we have multiple faces and average is reasonable, accept
            if len(similarities) >= 2 and avg_similarity >= (similarity_threshold * 0.8):
                logger.info(f"Face validation passed - average similarity: {avg_similarity:.3f} across {len(similarities)} faces")
                return True
            
            # Strategy 3: Check if this could be the same person with different conditions
            # Accept if similarity is above a lower threshold (0.4) for same person variations
            if max_similarity >= 0.4:
                logger.info(f"Face validation passed - likely same person with variations: {max_similarity:.3f}")
                return True
            
            logger.warning(f"Face validation failed - max: {max_similarity:.3f}, avg: {avg_similarity:.3f}, threshold: {similarity_threshold}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to validate face ownership: {e}")
            return False
        finally:
            session.close()

    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
