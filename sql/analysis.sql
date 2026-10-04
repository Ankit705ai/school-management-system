-- These queries use the students_cleaned table imported from data/students_cleaned.csv.
-- The current dataset has one table, so there is no meaningful JOIN to perform.
-- final_exam_score is intentionally not used to identify or explain at_risk students.

-- 1. How many cleaned student records are available?
SELECT COUNT(*) AS total_students
FROM students_cleaned;

-- 2. What is the overall risk distribution and percentage?
SELECT
	at_risk,
	COUNT(*) AS student_count,
	ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM students_cleaned), 2) AS percentage
FROM students_cleaned
GROUP BY at_risk
ORDER BY at_risk;

-- 3. How do the main numeric features differ by risk group?
SELECT
	at_risk,
	COUNT(*) AS student_count,
	ROUND(AVG(attendance_rate), 2) AS average_attendance_rate,
	ROUND(AVG(study_hours_per_week), 2) AS average_study_hours,
	ROUND(AVG(previous_exam_score), 2) AS average_previous_exam_score,
	ROUND(AVG(assignment_score), 2) AS average_assignment_score
FROM students_cleaned
GROUP BY at_risk
ORDER BY at_risk;

-- 4. What are the at-risk minus not-at-risk mean differences for key features?
SELECT
	ROUND(AVG(CASE WHEN at_risk = 1 THEN attendance_rate END) - AVG(CASE WHEN at_risk = 0 THEN attendance_rate END), 2) AS attendance_difference,
	ROUND(AVG(CASE WHEN at_risk = 1 THEN study_hours_per_week END) - AVG(CASE WHEN at_risk = 0 THEN study_hours_per_week END), 2) AS study_hours_difference,
	ROUND(AVG(CASE WHEN at_risk = 1 THEN previous_exam_score END) - AVG(CASE WHEN at_risk = 0 THEN previous_exam_score END), 2) AS previous_score_difference,
	ROUND(AVG(CASE WHEN at_risk = 1 THEN assignment_score END) - AVG(CASE WHEN at_risk = 0 THEN assignment_score END), 2) AS assignment_difference
FROM students_cleaned;

-- 5. What percentage of students are at risk in each program?
-- program is the available class-like grouping in this dataset.
SELECT
	program,
	COUNT(*) AS student_count,
	ROUND(100.0 * SUM(at_risk = 1) / COUNT(*), 2) AS at_risk_percentage
FROM students_cleaned
GROUP BY program
ORDER BY at_risk_percentage DESC;

-- 6. Which students have attendance below 70 percent?
SELECT
	student_id,
	program,
	attendance_rate,
	at_risk
FROM students_cleaned
WHERE attendance_rate < 70
ORDER BY attendance_rate;

-- 7. Which students have a previous exam score below 50?
SELECT
	student_id,
	program,
	previous_exam_score,
	at_risk
FROM students_cleaned
WHERE previous_exam_score < 50
ORDER BY previous_exam_score;

-- 8. Which students meet several pre-outcome high-risk conditions?
-- These conditions do not use final_exam_score, avoiding target leakage.
SELECT
	student_id,
	program,
	attendance_rate,
	study_hours_per_week,
	previous_exam_score,
	assignment_score,
	academic_support,
	at_risk
FROM students_cleaned
WHERE attendance_rate < 70
  AND study_hours_per_week < 8
  AND (previous_exam_score < 50 OR assignment_score < 60)
ORDER BY attendance_rate, previous_exam_score;

-- 9. Which available student groups have the highest risk percentage?
-- Combining program and academic support creates meaningful groups without a fake JOIN.
SELECT
	program,
	academic_support,
	COUNT(*) AS student_count,
	SUM(at_risk = 1) AS at_risk_count,
	ROUND(100.0 * SUM(at_risk = 1) / COUNT(*), 2) AS at_risk_percentage
FROM students_cleaned
GROUP BY program, academic_support
HAVING COUNT(*) >= 50
ORDER BY at_risk_percentage DESC;

-- 10. JOIN check: no JOIN is included because the project currently has one table
-- and no second entity table with a meaningful relationship to students_cleaned.
