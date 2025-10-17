#!/usr/bin/env python3
"""
Migration script to add multi-face support to the face embedding system.
"""
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app.services.database_service import DatabaseService
    from sqlalchemy import text
    
    def apply_migration():
        """Apply the multiple faces migration."""
        print("Starting migration to support multiple faces per user...")
        
        # Initialize database service
        db_service = DatabaseService()
        session = db_service.Session()
        
        try:
            # Check if app_id column exists
            result = session.execute(text("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name = 'face_records' AND column_name = 'app_id';
            """))
            
            if result.fetchone():
                print("Migration already applied - app_id column exists.")
                return True
            
            print("Applying migration...")
            
            # Add new columns
            print("Adding new columns...")
            session.execute(text("""
            ALTER TABLE face_records 
            ADD COLUMN IF NOT EXISTS app_id VARCHAR(100) DEFAULT 'default_app',
            ADD COLUMN IF NOT EXISTS face_alias VARCHAR(255),
            ADD COLUMN IF NOT EXISTS is_primary BOOLEAN DEFAULT FALSE;
            """))
            
            # Update existing records
            print("Updating existing records...")
            session.execute(text("""
            UPDATE face_records SET app_id = 'default_app' WHERE app_id IS NULL;
            """))
            
            # Make app_id NOT NULL
            session.execute(text("""
            ALTER TABLE face_records ALTER COLUMN app_id SET NOT NULL;
            """))
            
            # Create indexes
            print("Creating indexes...")
            session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_face_records_app_id ON face_records(app_id);
            """))
            session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_face_records_app_person ON face_records(app_id, person_id);
            """))
            session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_face_records_primary ON face_records(app_id, person_id, is_primary);
            """))
            
            session.commit()
            print("Migration completed successfully!")
            return True
            
        except Exception as e:
            print(f"Migration error: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    if __name__ == "__main__":
        success = apply_migration()
        sys.exit(0 if success else 1)
        
except Exception as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're in the project directory and all dependencies are installed.")
    sys.exit(1)
