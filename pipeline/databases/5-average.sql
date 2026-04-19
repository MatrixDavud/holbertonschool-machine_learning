-- This script computes the average score from the table second_table.
-- The result column is named average.
-- The database name is passed as an argument to the mysql command.

SELECT AVG(score) AS average FROM second_table;
