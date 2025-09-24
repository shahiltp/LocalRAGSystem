-- Create the rag_user with necessary permissions
CREATE USER rag_user WITH PASSWORD 'rag_password';
ALTER USER rag_user CREATEDB;
CREATE DATABASE rag_db OWNER rag_user;
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- Connect to the rag_db and create the vector extension
\c rag_db
CREATE EXTENSION IF NOT EXISTS vector;
