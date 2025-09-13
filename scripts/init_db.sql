-- Face Vector Embedding System Database Initialization
-- Complete setup from scratch

-- Create pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create face_records table
CREATE TABLE face_records (
    id SERIAL PRIMARY KEY,
    person_id VARCHAR(100) NOT NULL,
    person_name VARCHAR(255) NOT NULL,
    embedding vector(512) NOT NULL,  -- 512-dimensional vector for face embeddings
    confidence_score REAL DEFAULT 0.0,
    image_path VARCHAR(500),
    face_bbox TEXT,  -- JSON string storing bounding box coordinates
    landmarks TEXT,  -- JSON string storing facial landmarks
    encoding_model VARCHAR(50) DEFAULT 'facenet',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create face_processing_logs table
CREATE TABLE face_processing_logs (
    id SERIAL PRIMARY KEY,
    operation_type VARCHAR(50) NOT NULL,
    input_source VARCHAR(500),
    processing_time REAL,
    faces_detected INTEGER DEFAULT 0,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    metadata TEXT,  -- JSON string for additional data
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_face_records_person_id ON face_records(person_id);
CREATE INDEX idx_face_records_created_at ON face_records(created_at);
CREATE INDEX idx_face_records_is_active ON face_records(is_active);
CREATE INDEX idx_face_processing_logs_operation_type ON face_processing_logs(operation_type);
CREATE INDEX idx_face_processing_logs_created_at ON face_processing_logs(created_at);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update the updated_at column
CREATE TRIGGER update_face_records_updated_at
    BEFORE UPDATE ON face_records
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for active face records
CREATE VIEW active_face_records AS
SELECT * FROM face_records WHERE is_active = TRUE;

-- Create vector similarity search function
CREATE OR REPLACE FUNCTION find_similar_faces(
    query_embedding vector(128),
    similarity_threshold REAL DEFAULT 0.8,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    id INTEGER,
    person_id VARCHAR(100),
    person_name VARCHAR(255),
    similarity_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fr.id,
        fr.person_id,
        fr.person_name,
        1 - (fr.embedding <=> query_embedding) as similarity_score
    FROM face_records fr
    WHERE fr.is_active = TRUE
    AND 1 - (fr.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY fr.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Create user and grant permissions
CREATE USER admin WITH PASSWORD 'salvationboy@zuzu1';
GRANT ALL PRIVILEGES ON DATABASE face_db TO admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;

-- Insert initialization log
INSERT INTO face_processing_logs (operation_type, input_source, processing_time, success, metadata)
VALUES ('database_init', 'complete_setup', 0.0, TRUE, '{"message": "Database initialized successfully from scratch"}');

-- Display table structures for verification
\echo 'Database initialization complete!'
\echo 'Face Records Table:'
\d face_records
\echo 'Face Processing Logs Table:'
\d face_processing_logs

-- Show sample data
SELECT 'Initialization logs:' as info;
SELECT operation_type, success, metadata FROM face_processing_logs;
