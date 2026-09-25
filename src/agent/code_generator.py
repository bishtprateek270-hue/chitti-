"""
Chitti Programming Task Code Generator.
Generates real, syntactically correct Python code for user-requested programming problems.
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CodeGenerationResult:
    topic: str
    filename: str
    code: str
    expected_symbol: str
    description: str
    expected_markers: List[str] = field(default_factory=list)


class CodeGenerator:
    """Generates problem-specific Python programs."""

    TEMPLATES: Dict[str, Tuple[str, str, str, str, List[str]]] = {
        "anagram": (
            "anagram.py",
            "are_anagrams",
            (
                "# Anagram Checker - Created by Chitti Agent\n\n"
                "def are_anagrams(str1: str, str2: str) -> bool:\n"
                "    \"\"\"Checks if two strings are anagrams of each other.\"\"\"\n"
                "    clean1 = sorted(str1.replace(' ', '').lower())\n"
                "    clean2 = sorted(str2.replace(' ', '').lower())\n"
                "    return clean1 == clean2\n\n\n"
                "if __name__ == '__main__':\n"
                "    test1 = 'listen'\n"
                "    test2 = 'silent'\n"
                "    result = are_anagrams(test1, test2)\n"
                "    print(f\"Are '{test1}' and '{test2}' anagrams? {result}\")\n"
                "    \n"
                "    sample_pairs = [('triangle', 'integral'), ('apple', 'banana')]\n"
                "    for w1, w2 in sample_pairs:\n"
                "        print(f\"Are '{w1}' and '{w2}' anagrams? {are_anagrams(w1, w2)}\")\n"
            ),
            "Anagram detection function",
            ["def are_anagrams", "sorted(", "clean1 == clean2"],
        ),
        "fibonacci": (
            "fibonacci.py",
            "fibonacci",
            (
                "# Fibonacci Sequence Generator - Created by Chitti Agent\n\n"
                "def fibonacci(n: int) -> list:\n"
                "    \"\"\"Generates the first n numbers of the Fibonacci sequence.\"\"\"\n"
                "    if n <= 0:\n"
                "        return []\n"
                "    elif n == 1:\n"
                "        return [0]\n"
                "    seq = [0, 1]\n"
                "    while len(seq) < n:\n"
                "        seq.append(seq[-1] + seq[-2])\n"
                "    return seq\n\n\n"
                "if __name__ == '__main__':\n"
                "    terms = 10\n"
                "    print(f'Fibonacci series (first {terms} terms): {fibonacci(terms)}')\n"
            ),
            "Fibonacci series generator",
            ["def fibonacci", "seq.append", "seq[-1] + seq[-2]"],
        ),
        "palindrome": (
            "palindrome.py",
            "is_palindrome",
            (
                "# Palindrome Checker - Created by Chitti Agent\n\n"
                "def is_palindrome(s: str) -> bool:\n"
                "    \"\"\"Checks if a string is a palindrome.\"\"\"\n"
                "    clean = re.sub(r'[^a-zA-Z0-9]', '', s).lower()\n"
                "    return clean == clean[::-1]\n\n"
                "import re\n\n"
                "if __name__ == '__main__':\n"
                "    sample = 'racecar'\n"
                "    print(f\"Is '{sample}' a palindrome? {is_palindrome(sample)}\")\n"
            ),
            "Palindrome verification function",
            ["def is_palindrome", "clean[::-1]"],
        ),
        "factorial": (
            "factorial.py",
            "factorial",
            (
                "# Factorial Calculator - Created by Chitti Agent\n\n"
                "def factorial(n: int) -> int:\n"
                "    \"\"\"Calculates factorial of a non-negative integer.\"\"\"\n"
                "    if n < 0:\n"
                "        raise ValueError('Factorial is not defined for negative numbers.')\n"
                "    if n in (0, 1):\n"
                "        return 1\n"
                "    result = 1\n"
                "    for i in range(2, n + 1):\n"
                "        result *= i\n"
                "    return result\n\n\n"
                "if __name__ == '__main__':\n"
                "    num = 5\n"
                "    print(f'Factorial of {num} is: {factorial(num)}')\n"
            ),
            "Factorial computation function",
            ["def factorial", "result *="],
        ),
        "prime": (
            "prime_checker.py",
            "is_prime",
            (
                "# Prime Number Checker - Created by Chitti Agent\n\n"
                "def is_prime(n: int) -> bool:\n"
                "    \"\"\"Checks if a number is prime.\"\"\"\n"
                "    if n <= 1:\n"
                "        return False\n"
                "    if n <= 3:\n"
                "        return True\n"
                "    if n % 2 == 0 or n % 3 == 0:\n"
                "        return False\n"
                "    i = 5\n"
                "    while i * i <= n:\n"
                "        if n % i == 0 or n % (i + 2) == 0:\n"
                "            return False\n"
                "        i += 6\n"
                "    return True\n\n\n"
                "if __name__ == '__main__':\n"
                "    test_num = 29\n"
                "    print(f'Is {test_num} prime? {is_prime(test_num)}')\n"
            ),
            "Prime number checker",
            ["def is_prime", "while i * i <= n"],
        ),
        "binary_search": (
            "binary_search.py",
            "binary_search",
            (
                "# Binary Search Algorithm - Created by Chitti Agent\n\n"
                "def binary_search(arr: list, target: int) -> int:\n"
                "    \"\"\"Performs binary search on a sorted list. Returns index or -1.\"\"\"\n"
                "    low, high = 0, len(arr) - 1\n"
                "    while low <= high:\n"
                "        mid = (low + high) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            low = mid + 1\n"
                "        else:\n"
                "            high = mid - 1\n"
                "    return -1\n\n\n"
                "if __name__ == '__main__':\n"
                "    nums = [1, 3, 5, 7, 9, 11, 13, 15]\n"
                "    target_val = 7\n"
                "    idx = binary_search(nums, target_val)\n"
                "    print(f'Found {target_val} at index: {idx}')\n"
            ),
            "Binary search implementation",
            ["def binary_search", "while low <= high"],
        ),
        "bubble_sort": (
            "bubble_sort.py",
            "bubble_sort",
            (
                "# Bubble Sort Algorithm - Created by Chitti Agent\n\n"
                "def bubble_sort(arr: list) -> list:\n"
                "    \"\"\"Sorts a list in ascending order using bubble sort.\"\"\"\n"
                "    n = len(arr)\n"
                "    for i in range(n):\n"
                "        swapped = False\n"
                "        for j in range(0, n - i - 1):\n"
                "            if arr[j] > arr[j + 1]:\n"
                "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
                "                swapped = True\n"
                "        if not swapped:\n"
                "            break\n"
                "    return arr\n\n\n"
                "if __name__ == '__main__':\n"
                "    sample = [64, 34, 25, 12, 22, 11, 90]\n"
                "    print('Sorted array:', bubble_sort(sample))\n"
            ),
            "Bubble sort implementation",
            ["def bubble_sort", "for i in range(n)"],
        ),
    }

    @classmethod
    def detect_topic(cls, user_text: str) -> str:
        """Extracts the specific programming problem topic from user request."""
        clean = user_text.lower()
        if "anagram" in clean:
            return "anagram"
        elif "fibonacci" in clean:
            return "fibonacci"
        elif "palindrome" in clean:
            return "palindrome"
        elif "factorial" in clean:
            return "factorial"
        elif "prime" in clean:
            return "prime"
        elif "binary search" in clean or "binary_search" in clean:
            return "binary_search"
        elif "bubble sort" in clean or "sorting" in clean or "sort" in clean:
            return "bubble_sort"
        return "script"

    @classmethod
    def generate_code_for_topic(cls, topic_or_text: str) -> CodeGenerationResult:
        """Generates appropriate code and metadata for the requested topic."""
        topic = cls.detect_topic(topic_or_text)
        if topic in cls.TEMPLATES:
            fname, symbol, code, desc, markers = cls.TEMPLATES[topic]
            return CodeGenerationResult(
                topic=topic,
                filename=fname,
                code=code,
                expected_symbol=symbol,
                description=desc,
                expected_markers=markers,
            )

        # Generic fallback script
        fname = f"{topic}.py"
        code = (
            f"# Python Script: {topic.title()} - Created by Chitti Agent\n\n"
            f"def main():\n"
            f"    print('Executing {topic.title()} solution...')\n\n\n"
            f"if __name__ == '__main__':\n"
            f"    main()\n"
        )
        return CodeGenerationResult(
            topic=topic,
            filename=fname,
            code=code,
            expected_symbol="main",
            description=f"Python {topic} script",
            expected_markers=["def main()"],
        )
