"""
PostgreSQL database schema and queries for vehicle classification system.
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
import asyncpg
from contextlib import asynccontextmanager


class Database:
    """PostgreSQL database manager."""
    
    def __init__(self, database_url: str):
        """
        Initialize database connection.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Create database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
        await self.create_tables()
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def create_tables(self):
        """Create database tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS classifications (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    image_path TEXT,
                    vehicle_type TEXT,
                    axle_count INTEGER,
                    predicted_class TEXT,
                    corrected_class TEXT,
                    confidence FLOAT,
                    processing_time FLOAT,
                    bbox_x1 INTEGER,
                    bbox_y1 INTEGER,
                    bbox_x2 INTEGER,
                    bbox_y2 INTEGER
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    classification_id UUID REFERENCES classifications(id) ON DELETE CASCADE,
                    corrected_class TEXT,
                    corrected_axles INTEGER,
                    notes TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes for better query performance
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_classifications_timestamp 
                ON classifications(timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_classification_id 
                ON feedback(classification_id)
            """)
    
    async def insert_classification(
        self,
        image_path: str,
        vehicle_type: str,
        axle_count: int,
        predicted_class: str,
        confidence: float,
        processing_time: float,
        bbox: Optional[Dict] = None
    ) -> str:
        """
        Insert a new classification record.
        
        Returns:
            UUID of inserted record
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO classifications (
                    image_path, vehicle_type, axle_count, predicted_class,
                    confidence, processing_time, bbox_x1, bbox_y1, bbox_x2, bbox_y2
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """,
                image_path,
                vehicle_type,
                axle_count,
                predicted_class,
                confidence,
                processing_time,
                bbox.get("x1") if bbox else None,
                bbox.get("y1") if bbox else None,
                bbox.get("x2") if bbox else None,
                bbox.get("y2") if bbox else None
            )
            return str(row["id"])
    
    async def insert_feedback(
        self,
        classification_id: str,
        corrected_class: str,
        corrected_axles: Optional[int] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Insert feedback for a classification.
        
        Returns:
            UUID of feedback record
        """
        async with self.pool.acquire() as conn:
            # Update corrected_class in classifications table
            await conn.execute("""
                UPDATE classifications
                SET corrected_class = $1
                WHERE id = $2
            """, corrected_class, classification_id)
            
            # Insert feedback record
            row = await conn.fetchrow("""
                INSERT INTO feedback (classification_id, corrected_class, corrected_axles, notes)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, classification_id, corrected_class, corrected_axles, notes)
            
            return str(row["id"])
    
    async def get_classifications(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """Get paginated classifications."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT 
                    c.id,
                    c.timestamp,
                    c.image_path,
                    c.vehicle_type,
                    c.axle_count,
                    c.predicted_class,
                    c.corrected_class,
                    c.confidence,
                    c.processing_time,
                    c.bbox_x1,
                    c.bbox_y1,
                    c.bbox_x2,
                    c.bbox_y2,
                    f.corrected_axles,
                    f.notes
                FROM classifications c
                LEFT JOIN feedback f ON c.id = f.classification_id
                ORDER BY c.timestamp DESC
                LIMIT $1 OFFSET $2
            """, limit, offset)
            
            return [dict(row) for row in rows]
    
    async def get_classification_by_id(self, classification_id: str) -> Optional[Dict]:
        """Get a specific classification by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 
                    c.*,
                    f.corrected_axles,
                    f.notes
                FROM classifications c
                LEFT JOIN feedback f ON c.id = f.classification_id
                WHERE c.id = $1
            """, classification_id)
            
            return dict(row) if row else None
    
    async def get_accuracy_metrics(self) -> Dict:
        """
        Calculate accuracy metrics based on feedback.
        
        Returns:
            Dictionary with accuracy statistics
        """
        async with self.pool.acquire() as conn:
            # Overall accuracy
            overall = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_with_feedback,
                    SUM(CASE WHEN predicted_class = corrected_class THEN 1 ELSE 0 END) as correct
                FROM classifications
                WHERE corrected_class IS NOT NULL
            """)
            
            total = overall["total_with_feedback"] or 0
            correct = overall["correct"] or 0
            accuracy = (correct / total * 100) if total > 0 else 0
            
            # Per-class accuracy
            per_class = await conn.fetch("""
                SELECT 
                    predicted_class,
                    COUNT(*) as total,
                    SUM(CASE WHEN predicted_class = corrected_class THEN 1 ELSE 0 END) as correct
                FROM classifications
                WHERE corrected_class IS NOT NULL
                GROUP BY predicted_class
            """)
            
            # Confusion matrix
            confusion = await conn.fetch("""
                SELECT 
                    predicted_class,
                    corrected_class,
                    COUNT(*) as count
                FROM classifications
                WHERE corrected_class IS NOT NULL
                GROUP BY predicted_class, corrected_class
            """)
            
            # Total classifications (including those without feedback)
            total_classifications = await conn.fetchval("""
                SELECT COUNT(*) FROM classifications
            """)
            
            return {
                "overall_accuracy": accuracy,
                "total_with_feedback": total,
                "total_correct": correct,
                "total_classifications": total_classifications,
                "per_class": [
                    {
                        "class": row["predicted_class"],
                        "total": row["total"],
                        "correct": row["correct"],
                        "accuracy": (row["correct"] / row["total"] * 100) if row["total"] > 0 else 0
                    }
                    for row in per_class
                ],
                "confusion_matrix": [
                    {
                        "predicted": row["predicted_class"],
                        "actual": row["corrected_class"],
                        "count": row["count"]
                    }
                    for row in confusion
                ]
            }
    
    async def clear_all_data(self):
        """Delete all data from the database."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM feedback")
            await conn.execute("DELETE FROM classifications")
    
    async def get_statistics(self) -> Dict:
        """Get general statistics."""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_classifications,
                    COUNT(CASE WHEN predicted_class = 'Class 1' THEN 1 END) as class_1_count,
                    COUNT(CASE WHEN predicted_class = 'Class 2' THEN 1 END) as class_2_count,
                    COUNT(CASE WHEN predicted_class = 'Class 3' THEN 1 END) as class_3_count,
                    COUNT(CASE WHEN predicted_class = 'Class 4' THEN 1 END) as class_4_count,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time) as avg_processing_time
                FROM classifications
            """)
            
            return dict(stats)


# Global database instance
db: Optional[Database] = None


async def get_database() -> Database:
    """Get database instance."""
    global db
    if db is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        db = Database(database_url)
        await db.connect()
    return db

