-- This script lists all shows without a genre linked.
-- Each record displays: tv_shows.title - tv_show_genres.genre_id.
-- Results are sorted by title and genre_id in ascending order.

SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
LEFT JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
WHERE tv_show_genres.genre_id IS NULL
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
