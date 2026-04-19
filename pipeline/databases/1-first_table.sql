-- This script creates the table first_table in the current database.
-- It does not fail if the table already exists.
-- No SELECT or SHOW statements are used.

CREATE TABLE IF NOT EXISTS first_table (
    id INT,
    name VARCHAR(256)
);
