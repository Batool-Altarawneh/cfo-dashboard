-- init.sql
-- This script runs automatically the first time the PostgreSQL container starts.
-- It prepares the database structure needed for the CFO Financial KPI Dashboard project.
--
-- In this project, I separate raw data from clean reporting tables using schemas:
-- 1. staging    -> raw data loaded from CSV files
-- 2. production -> clean star schema tables used by Power BI
--
-- I also create a separate database for Airflow:
-- airflow_metadata -> Airflow internal database for DAG runs, task states, users, and logs

-- Important note:
-- This file only runs automatically when the PostgreSQL database is created for the first time.
-- If the Docker volume already exists, PostgreSQL will not re-run this script automatically.


-- ---------------------------------------------------------------------------
-- Create staging schema
-- ---------------------------------------------------------------------------
-- The staging schema is used as the raw landing area.
-- Data loaded here is close to the original CSV format.
-- At this stage, the data may still contain duplicates, formatting issues,
-- missing values, or columns that need transformation.
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS staging;


-- ---------------------------------------------------------------------------
-- Create production schema
-- ---------------------------------------------------------------------------
-- The production schema is used for clean and validated reporting tables.
-- This is where the final star schema will live
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS production;

-- ---------------------------------------------------------------------------
-- Create Airflow metadata database
-- ---------------------------------------------------------------------------
-- Airflow needs its own database to track:
-- - DAG runs
-- - task states
-- - schedules
-- - users
-- - logs metadata
--
-- I keep this separate from cfo_dashboard so Airflow internal tables do not mix with my project data tables.
--
-- Important:
-- PostgreSQL does not support: CREATE DATABASE IF NOT EXISTS
--
-- But this init.sql file runs only once when the Docker volume is first created,so this is okay for the first setup.
-- ---------------------------------------------------------------------------

CREATE DATABASE airflow_metadata;


-- ---------------------------------------------------------------------------
-- Confirmation query
-- ---------------------------------------------------------------------------
-- This query checks that the two schemas were created successfully.
-- It reads from PostgreSQL's metadata table called information_schema.schemata.
-- ---------------------------------------------------------------------------

SELECT schema_name
FROM information_schema.schemata
WHERE schema_name IN ('staging', 'production');

-- ---------------------------------------------------------------------------
-- Confirmation query for Airflow metadata database
-- ---------------------------------------------------------------------------
-- This checks that the airflow_metadata database exists.
-- pg_database is PostgreSQL's internal table that lists databases.
-- ---------------------------------------------------------------------------

SELECT datname
FROM pg_database
WHERE datname = 'airflow_metadata';