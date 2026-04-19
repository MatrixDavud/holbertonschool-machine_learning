-- This script creates a SQL function SafeDiv that safely divides two integers.
-- It returns a / b if b is not zero, otherwise returns 0.

DELIMITER $$

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS INT
DETERMINISTIC
BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a DIV b;
    END IF;
END$$

DELIMITER ;
