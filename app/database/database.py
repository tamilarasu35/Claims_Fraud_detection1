"""
SQLite Database Connection and Initialization Manager.
Supports clean schema creation and thread-safe connections.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Generator
from app.config.settings import settings
from app.utils.logger import logger

_local = threading.local()

def get_db_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection with WAL mode enabled."""
    if not hasattr(_local, "connection") or _local.connection is None:
        db_path = settings.DATABASE_PATH
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Optimize SQLite performance & concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        
        _local.connection = conn
    return _local.connection

def close_db_connection():
    """Close the thread-local SQLite connection if open."""
    if hasattr(_local, "connection") and _local.connection is not None:
        _local.connection.close()
        _local.connection = None

def init_db():
    """Initialize all required SQLite tables and default seeds."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    logger.info(f"Initializing database at: {settings.DATABASE_PATH}")
    
    # 1. Roles table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 2. Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        email TEXT,
        role_name TEXT NOT NULL DEFAULT 'USER',
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        FOREIGN KEY (role_name) REFERENCES roles(name)
    );
    """)
    
    # 3. Datasets table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        record_count INTEGER DEFAULT 0,
        provider_count INTEGER DEFAULT 0,
        uploaded_by TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 4. Model Versions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version_tag TEXT UNIQUE NOT NULL,
        xgboost_path TEXT NOT NULL,
        ebm_path TEXT NOT NULL,
        metrics_json TEXT,
        is_active BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 5. Analysis Runs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_uuid TEXT UNIQUE NOT NULL,
        dataset_id INTEGER,
        initiated_by TEXT NOT NULL,
        total_providers INTEGER DEFAULT 0,
        flagged_fraudulent INTEGER DEFAULT 0,
        model_version TEXT,
        status TEXT DEFAULT 'COMPLETED',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    );
    """)
    
    # 6. Providers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS providers (
        provider_id TEXT PRIMARY KEY,
        total_claims INTEGER DEFAULT 0,
        total_reimbursement REAL DEFAULT 0.0,
        beneficiary_count INTEGER DEFAULT 0,
        inpatient_claims INTEGER DEFAULT 0,
        outpatient_claims INTEGER DEFAULT 0,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # 7. Fraud Results table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fraud_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_uuid TEXT NOT NULL,
        provider_id TEXT NOT NULL,
        classification TEXT NOT NULL,
        fraud_probability REAL NOT NULL,
        risk_score INTEGER NOT NULL,
        risk_level TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        top_features_json TEXT,
        negotiation_argument TEXT,
        negotiation_challenge TEXT,
        arbitrator_reasoning TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (provider_id) REFERENCES providers(provider_id)
    );
    """)
    
    # 8. Investigations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS investigations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider_id TEXT NOT NULL,
        investigator_username TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'NEW',
        priority TEXT DEFAULT 'HIGH',
        notes TEXT,
        decision TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (provider_id) REFERENCES providers(provider_id)
    );
    """)
    
    # 9. Audit Logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        username TEXT NOT NULL,
        role TEXT NOT NULL,
        action TEXT NOT NULL,
        target_resource TEXT,
        details TEXT,
        ip_address TEXT DEFAULT '127.0.0.1'
    );
    """)
    
    # Seed default roles if empty
    default_roles = [
        ('USER', 'Base user who can upload claims & initiate fraud analysis'),
        ('INVESTIGATOR', 'Fraud Investigator who analyzes provider drill-downs and case files'),
        ('MANAGER', 'Manager who monitors executive fraud trends & aggregate stats'),
        ('ADMIN', 'System Administrator with full access to user, model & config management')
    ]
    cursor.executemany("""
    INSERT OR IGNORE INTO roles (name, description) VALUES (?, ?);
    """, default_roles)
    
    conn.commit()
    logger.info("Database initialized successfully with core schema and default roles.")
