from pathlib import Path
import random

class MathDatasetGenerator:
    def __init__(self, file_path, num_questions=100):
        self.file_path = file_path
        self.num_questions = num_questions

    def generate(self):
        operations = ['*', '//', '**']  # Added exponentiation for complexity
        with open(self.file_path, 'w') as f:
            for _ in range(self.num_questions):
                a = random.randint(100, 1000)  # Increased range for larger numbers
                b = random.randint(100, 1000)
                operation = random.choice(operations)
                if operation == '//':
                    # Ensure no division by zero and integer division
                    b = random.randint(1, 100)
                    answer = a // b
                elif operation == '**':
                    # Limit exponentiation to avoid overly large results
                    b = 2
                    answer = a ** b
                else:
                    answer = eval(f"{a} {operation} {b}")
                f.write(f"{a}{operation}{b}, {answer}\n")

if __name__ == "__main__":
    generator = MathDatasetGenerator(Path(__file__).parent / "dataset" / "test_dataset.txt", num_questions=1000)
    generator.generate() 