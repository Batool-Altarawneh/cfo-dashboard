-- init.sql
-- This script runs automatically the first time the PostgreSQL container starts.
-- It prepares the database structure needed for the CFO Financial KPI Dashboard project.
--
-- In this project, I separate raw data from clean reporting tables using schemas:
-- 1. staging    -> raw data loaded from CSV files
-- 2. production -> clean star schema tables used by Power BI
--
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
-- Confirmation query
-- ---------------------------------------------------------------------------
-- This query checks that the two schemas were created successfully.
-- It reads from PostgreSQL's metadata table called information_schema.schemata.
-- ---------------------------------------------------------------------------

SELECT schema_name
FROM information_schema.schemata
WHERE schema_name IN ('staging', 'production');