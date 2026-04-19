-- This script lists all records from second_table with a score >= 10.
-- Results display score and name, ordered by score descending.

SELECT score, name
FROM second_table
WHERE score >= 10
ORDER BY score DESC;
