-- Migration script to support multiple faces per user and app isolation
-- This script adds app_id support and face management features

-- Add new columns to face_records table
ALTER TABLE face_records 
ADD COLUMN IF NOT EXISTS app_id VARCHAR(100) DEFAULT 'default_app',
ADD COLUMN IF NOT EXISTS face_alias VARCHAR(255),
ADD COLUMN IF NOT EXISTS is_primary BOOLEAN DEFAULT FALSE;

-- Make app_id NOT NULL after setting default values
UPDATE face_records SET app_id = 'default_app' WHERE app_id IS NULL;
ALTER TABLE face_records ALTER COLUMN app_id SET NOT NULL;

-- Create new indexes for better performance
CREATE INDEX IF NOT EXISTS idx_face_records_app_id ON face_records(app_id);
CREATE INDEX IF NOT EXISTS idx_face_records_app_person ON face_records(app_id, person_id);
CREATE INDEX IF NOT EXISTS idx_face_records_primary ON face_records(app_id, person_id, is_primary);

-- Update the vector similarity search function to support app filtering
CREATE OR REPLACE FUNCTION find_similar_faces_by_app(
    app_id_param VARCHAR(100),
    query_embedding vector(512),
    similarity_threshold REAL DEFAULT 0.8,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE(
    id INTEGER,
    person_id VARCHAR(100),
    person_name VARCHAR(255),
    face_alias VARCHAR(255),
    similarity_score REAL,
    is_primary BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fr.id,
        fr.person_id,
        fr.person_name,
        fr.face_alias,
        1 - (fr.embedding <=> query_embedding) as similarity_score,
        fr.is_primary
    FROM face_records fr
    WHERE fr.is_active = TRUE
    AND fr.app_id = app_id_param
    AND 1 - (fr.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY fr.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Create function to get best match per person
CREATE OR REPLACE FUNCTION find_best_match_per_person(
    app_id_param VARCHAR(100),
    query_embedding vector(512),
    similarity_threshold REAL DEFAULT 0.8
)
RETURNS TABLE(
    person_id VARCHAR(100),
    person_name VARCHAR(255),
    face_record_id INTEGER,
    face_alias VARCHAR(255),
    similarity_score REAL,
    is_primary BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH ranked_matches AS (
        SELECT 
            fr.person_id,
            fr.person_name,
            fr.id as face_record_id,
            fr.face_alias,
            1 - (fr.embedding <=> query_embedding) as similarity_score,
            fr.is_primary,
            ROW_NUMBER() OVER (
                PARTITION BY fr.person_id 
                ORDER BY 1 - (fr.embedding <=> query_embedding) DESC
            ) as rank
        FROM face_records fr
        WHERE fr.is_active = TRUE
        AND fr.app_id = app_id_param
        AND 1 - (fr.embedding <=> query_embedding) >= similarity_threshold
    )
    SELECT 
        rm.person_id,
        rm.person_name,
        rm.face_record_id,
        rm.face_alias,
        rm.similarity_score,
        rm.is_primary
    FROM ranked_matches rm
    WHERE rm.rank = 1
    ORDER BY rm.similarity_score DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to ensure only one primary face per person
CREATE OR REPLACE FUNCTION ensure_single_primary_face()
RETURNS TRIGGER AS $$
BEGIN
    -- If setting a face as primary, unset other primary faces for the same person
    IF NEW.is_primary = TRUE THEN
        UPDATE face_records 
        SET is_primary = FALSE 
        WHERE app_id = NEW.app_id 
        AND person_id = NEW.person_id 
        AND id != NEW.id 
        AND is_primary = TRUE;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for primary face management
DROP TRIGGER IF EXISTS ensure_single_primary_face_trigger ON face_records;
CREATE TRIGGER ensure_single_primary_face_trigger
    BEFORE INSERT OR UPDATE ON face_records
    FOR EACH ROW
    EXECUTE FUNCTION ensure_single_primary_face();

-- Create view for person face counts by app
CREATE OR REPLACE VIEW person_face_counts AS
SELECT 
    app_id,
    person_id,
    person_name,
    COUNT(*) as face_count,
    COUNT(CASE WHEN is_primary THEN 1 END) as primary_count,
    MAX(created_at) as last_face_added
FROM face_records 
WHERE is_active = TRUE
GROUP BY app_id, person_id, person_name;

-- Create view for active faces with enhanced info
CREATE OR REPLACE VIEW active_faces_enhanced AS
SELECT 
    fr.*,
    pfc.face_count,
    pfc.primary_count
FROM face_records fr
JOIN person_face_counts pfc ON fr.app_id = pfc.app_id AND fr.person_id = pfc.person_id
WHERE fr.is_active = TRUE;

-- Insert migration log
INSERT INTO face_processing_logs (operation_type, input_source, processing_time, success, metadata)
VALUES ('database_migration', 'multiple_faces_support', 0.0, TRUE, 
        '{"message": "Added multiple faces per user and app isolation support"}');

-- Display migration summary
SELECT 
    'Migration Complete' as status,
    COUNT(*) as total_face_records,
    COUNT(DISTINCT app_id) as unique_apps,
    COUNT(DISTINCT CONCAT(app_id, '-', person_id)) as unique_persons
FROM face_records WHERE is_active = TRUE;
