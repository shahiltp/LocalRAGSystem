#!/usr/bin/env python3
"""
Database Health Check Utility
Provides comprehensive database validation and health checking functions
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine, text
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class DatabaseChecker:
    """Comprehensive database health checker for PostgreSQL with pgvector"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database checker
        
        Args:
            database_url: PostgreSQL connection URL. If None, uses DATABASE_URL env var
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL must be provided or set as environment variable")
        
        self.engine = create_engine(self.database_url)
        self.db_info = self._parse_database_url()
    
    def _parse_database_url(self) -> Dict[str, str]:
        """Parse database URL to extract connection details"""
        parsed = urlparse(self.database_url)
        return {
            "host": parsed.hostname,
            "port": parsed.port,
            "database": parsed.path.lstrip('/'),
            "user": parsed.username,
            "password": "***" if parsed.password else None
        }
    
    def check_connection(self) -> Tuple[bool, str]:
        """
        Test basic database connectivity
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                return True, f"Connected to PostgreSQL: {version[:50]}..."
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def check_extensions(self) -> Dict[str, bool]:
        """
        Check if required extensions are installed
        
        Returns:
            Dict mapping extension names to their installation status
        """
        extensions = {}
        try:
            with self.engine.connect() as conn:
                # Check pgvector extension
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'vector'
                    );
                """))
                extensions['vector'] = result.fetchone()[0]
                
                # Check other useful extensions
                result = conn.execute(text("""
                    SELECT extname FROM pg_extension 
                    WHERE extname IN ('uuid-ossp', 'pg_trgm', 'btree_gin')
                    ORDER BY extname;
                """))
                installed_exts = [row[0] for row in result.fetchall()]
                extensions['uuid-ossp'] = 'uuid-ossp' in installed_exts
                extensions['pg_trgm'] = 'pg_trgm' in installed_exts
                extensions['btree_gin'] = 'btree_gin' in installed_exts
                
        except Exception as e:
            logger.error(f"Error checking extensions: {e}")
            extensions['error'] = str(e)
        
        return extensions
    
    def check_tables(self) -> Dict[str, Dict[str, any]]:
        """
        Check all tables in the public schema and their data
        
        Returns:
            Dict mapping table names to their status info
        """
        tables_info = {}
        try:
            with self.engine.connect() as conn:
                # Get all tables
                result = conn.execute(text("""
                    SELECT table_name, table_type 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """))
                tables = result.fetchall()
                
                for table_name, table_type in tables:
                    table_info = {
                        'type': table_type,
                        'exists': True,
                        'row_count': 0,
                        'has_data': False,
                        'columns': [],
                        'issues': []
                    }
                    
                    try:
                        # Get row count
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        table_info['row_count'] = result.fetchone()[0]
                        table_info['has_data'] = table_info['row_count'] > 0
                        
                        # Get column info
                        result = conn.execute(text(f"""
                            SELECT column_name, data_type, is_nullable
                            FROM information_schema.columns 
                            WHERE table_name = '{table_name}'
                            ORDER BY ordinal_position;
                        """))
                        table_info['columns'] = [
                            {'name': row[0], 'type': row[1], 'nullable': row[2]}
                            for row in result.fetchall()
                        ]
                        
                        # Check for common issues
                        if table_info['row_count'] == 0:
                            table_info['issues'].append("Table is empty")
                        
                        # Check for null values in key columns
                        for col in table_info['columns']:
                            if col['name'] in ['embedding', 'text', 'content']:
                                result = conn.execute(text(f"""
                                    SELECT COUNT(*) FROM {table_name} 
                                    WHERE {col['name']} IS NULL
                                """))
                                null_count = result.fetchone()[0]
                                if null_count > 0:
                                    table_info['issues'].append(f"{null_count} null values in {col['name']}")
                        
                    except Exception as e:
                        table_info['issues'].append(f"Error checking table: {str(e)}")
                    
                    tables_info[table_name] = table_info
                    
        except Exception as e:
            logger.error(f"Error checking tables: {e}")
            tables_info['error'] = str(e)
        
        return tables_info
    
    def check_rag_tables(self) -> Dict[str, any]:
        """
        Specifically check RAG-related tables for data quality
        
        Returns:
            Dict with RAG table validation results
        """
        rag_info = {
            'vector_table': None,
            'document_table': None,
            'overall_status': 'unknown',
            'total_documents': 0,
            'total_embeddings': 0,
            'issues': []
        }
        
        try:
            with self.engine.connect() as conn:
                # Check for common RAG table names
                rag_tables = ['data_data_document_embeddings', 'data_document_text_store', 'data_doc_md_contextual_20250830']
                
                for table_name in rag_tables:
                    try:
                        result = conn.execute(text(f"""
                            SELECT EXISTS (
                                SELECT 1 FROM information_schema.tables 
                                WHERE table_name = '{table_name}'
                            );
                        """))
                        table_exists = result.fetchone()[0]
                        
                        if table_exists:
                            # Get row count
                            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                            count = result.fetchone()[0]
                            
                            if 'embedding' in table_name.lower():
                                rag_info['vector_table'] = {
                                    'name': table_name,
                                    'count': count,
                                    'has_data': count > 0
                                }
                                rag_info['total_embeddings'] = count
                                
                                # Check embedding quality
                                result = conn.execute(text(f"""
                                    SELECT COUNT(*) FROM {table_name} 
                                    WHERE embedding IS NOT NULL
                                """))
                                valid_embeddings = result.fetchone()[0]
                                if valid_embeddings < count:
                                    rag_info['issues'].append(f"{count - valid_embeddings} null embeddings in {table_name}")
                            
                            if 'text' in table_name.lower() or 'document' in table_name.lower():
                                rag_info['document_table'] = {
                                    'name': table_name,
                                    'count': count,
                                    'has_data': count > 0
                                }
                                rag_info['total_documents'] = count
                                
                                # Check text quality
                                result = conn.execute(text(f"""
                                    SELECT COUNT(*) FROM {table_name} 
                                    WHERE text IS NOT NULL AND text != ''
                                """))
                                valid_text = result.fetchone()[0]
                                if valid_text < count:
                                    rag_info['issues'].append(f"{count - valid_text} empty text entries in {table_name}")
                    
                    except Exception as e:
                        rag_info['issues'].append(f"Error checking {table_name}: {str(e)}")
                
                # Determine overall status
                if rag_info['total_documents'] > 0 and rag_info['total_embeddings'] > 0:
                    rag_info['overall_status'] = 'healthy'
                elif rag_info['total_documents'] > 0 or rag_info['total_embeddings'] > 0:
                    rag_info['overall_status'] = 'partial'
                else:
                    rag_info['overall_status'] = 'empty'
                
        except Exception as e:
            rag_info['issues'].append(f"Database error: {str(e)}")
            rag_info['overall_status'] = 'error'
        
        return rag_info
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> List[Dict]:
        """
        Get sample data from a table for inspection
        
        Args:
            table_name: Name of the table to sample
            limit: Number of rows to return
            
        Returns:
            List of sample rows
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT * FROM {table_name} 
                    LIMIT {limit}
                """))
                
                columns = result.keys()
                samples = []
                for row in result.fetchall():
                    sample = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        # Truncate long text fields
                        if isinstance(value, str) and len(value) > 200:
                            value = value[:200] + "..."
                        sample[col] = value
                    samples.append(sample)
                
                return samples
                
        except Exception as e:
            logger.error(f"Error sampling {table_name}: {e}")
            return []
    
    def comprehensive_check(self) -> Dict[str, any]:
        """
        Perform a comprehensive database health check
        
        Returns:
            Complete health check report
        """
        report = {
            'timestamp': None,
            'database_info': self.db_info,
            'connection': {},
            'extensions': {},
            'tables': {},
            'rag_status': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        # Import datetime here to avoid issues if not available
        from datetime import datetime
        report['timestamp'] = datetime.now().isoformat()
        
        # Check connection
        success, message = self.check_connection()
        report['connection'] = {'success': success, 'message': message}
        
        if not success:
            report['overall_status'] = 'connection_failed'
            report['recommendations'].append("Fix database connection issues")
            return report
        
        # Check extensions
        report['extensions'] = self.check_extensions()
        
        # Check tables
        report['tables'] = self.check_tables()
        
        # Check RAG-specific status
        report['rag_status'] = self.check_rag_tables()
        
        # Determine overall status
        if report['rag_status']['overall_status'] == 'healthy':
            report['overall_status'] = 'healthy'
        elif report['rag_status']['overall_status'] == 'partial':
            report['overall_status'] = 'partial'
            report['recommendations'].append("Some RAG tables are missing data")
        elif report['rag_status']['overall_status'] == 'empty':
            report['overall_status'] = 'empty'
            report['recommendations'].append("Run data ingestion to populate tables")
        else:
            report['overall_status'] = 'error'
            report['recommendations'].append("Check database configuration and run ingestion")
        
        # Add extension recommendations
        if not report['extensions'].get('vector', False):
            report['recommendations'].append("Install pgvector extension: CREATE EXTENSION vector;")
        
        return report


# Convenience functions for easy use
def quick_database_check() -> bool:
    """Quick check if database is accessible and has data"""
    try:
        checker = DatabaseChecker()
        success, _ = checker.check_connection()
        if not success:
            return False
        
        rag_status = checker.check_rag_tables()
        return rag_status['overall_status'] in ['healthy', 'partial']
    except Exception as e:
        logger.error(f"Quick database check failed: {e}")
        return False


def get_database_status() -> Dict[str, any]:
    """Get comprehensive database status"""
    try:
        checker = DatabaseChecker()
        return checker.comprehensive_check()
    except Exception as e:
        return {
            'overall_status': 'error',
            'error': str(e),
            'recommendations': ['Check DATABASE_URL configuration']
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Database Health Check")
    print("=" * 50)
    
    try:
        checker = DatabaseChecker()
        report = checker.comprehensive_check()
        
        print(f"Overall Status: {report['overall_status']}")
        print(f"Connection: {report['connection']['message']}")
        print(f"RAG Status: {report['rag_status']['overall_status']}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
