📌 Project Title

Exam Timetable Optimization using Genetic Algorithm

📖 Description

This project solves the exam scheduling problem using a Genetic Algorithm approach.
The system assigns courses to rooms, time slots, and lecturers while minimizing conflicts and violations of constraints such as room capacity and student exam clashes.

The goal is to generate an optimized exam timetable that maximizes scheduling efficiency and fairness.

⚙️ Problem Statement

Scheduling exams manually is a complex combinatorial problem with many constraints:

Limited room capacities
Students enrolled in multiple courses
Time slot conflicts
Lecturer availability

This project automates the process using evolutionary computation.

🧠 Approach

A Genetic Algorithm is implemented with:

Chromosome representation: course scheduling assignments
Fitness function: penalty-based evaluation
Selection: Tournament selection
Crossover: Uniform / One-point crossover
Mutation: Random room/time slot changes
Constraint handling: penalty system
📊 Features
Generates optimized exam schedules
Handles multiple real-world constraints
Reduces student exam conflicts
Ensures room capacity compliance
Visual analysis of evolution progress
GUI interface using Tkinter
🛠️ Technologies Used
Python
Pandas
Matplotlib
Tkinter
Jupyter Notebook
📈 Output

The algorithm evolves over generations to improve timetable quality by reducing penalty scores and increasing fitness.
