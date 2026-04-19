-- This script displays the average temperature (Fahrenheit) by city.
-- Results are ordered by average temperature in descending order.

SELECT city, AVG(value) AS avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_temp DESC;
