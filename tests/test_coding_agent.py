"""
Unit tests for Chitti General-Purpose Coding Agent, Toolchains, and Code Validator.
"""

import pytest
from src.agent.code_generator import CodeGenerator
from src.agent.code_spec import ProgrammingTaskSpec, CodeFileSpec
from src.agent.code_validator import CodeValidator
from src.agent.toolchain import ToolchainManager


def test_task_parsing_arbitrary_languages():
    # 1. Python Anagram
    spec1 = CodeGenerator.parse_programming_task("Write an Anagram program in Python")
    assert spec1.language == "python"
    assert spec1.filename == "anagram.py"
    assert spec1.execution_requested is False

    # 2. C++ Fibonacci
    spec2 = CodeGenerator.parse_programming_task("Write a Fibonacci program in C++")
    assert spec2.language == "cpp"
    assert spec2.filename == "fibonacci.cpp"
    assert spec2.execution_requested is False

    # 3. Java REST API
    spec3 = CodeGenerator.parse_programming_task("Create a REST API in Java")
    assert spec3.language == "java"
    assert spec3.filename == "ApiController.java" or spec3.filename.endswith(".java")

    # 4. React Login Page
    spec4 = CodeGenerator.parse_programming_task("Create a React login page")
    assert spec4.language == "react"
    assert spec4.filename == "App.jsx"

    # 5. Rust Sorting
    spec5 = CodeGenerator.parse_programming_task("Write a sorting algorithm in Rust")
    assert spec5.language == "rust"
    assert spec5.filename == "sorting.rs"

    # 6. C Linked list
    spec6 = CodeGenerator.parse_programming_task("Make a C program for linked list operations")
    assert spec6.language == "c"
    assert spec6.filename == "linked_list.c"

    # 7. Hinglish Java calculator
    spec7 = CodeGenerator.parse_programming_task("Java mein calculator banao")
    assert spec7.language == "java"
    assert spec7.filename == "Calculator.java"


def test_task_parsing_execution_intent():
    # Only creation
    spec_create = CodeGenerator.parse_programming_task("Python mein calculator banao")
    assert spec_create.execution_requested is False

    # Creation + execution
    spec_exec = CodeGenerator.parse_programming_task("Python mein calculator banao aur run karo")
    assert spec_exec.execution_requested is True

    spec_exec2 = CodeGenerator.parse_programming_task("C++ me linked list implement karo aur execute karo")
    assert spec_exec2.execution_requested is True


def test_code_validator_syntax_and_placeholders():
    # 1. Valid Python
    val_py = CodeValidator.validate_code(
        "def add(a, b):\n    return a + b\n\nif __name__ == '__main__':\n    print(add(2, 3))\n",
        language="python",
    )
    assert val_py.valid is True
    assert len(val_py.issues) == 0

    # 2. Placeholder rejection
    val_lazy = CodeValidator.validate_code(
        "def compute():\n    # TODO: implement logic\n    pass\n",
        language="python",
    )
    assert val_lazy.valid is False
    assert any("placeholder" in issue.lower() or "empty body" in issue.lower() for issue in val_lazy.issues)

    # 3. C++ bracket mismatch
    val_cpp_bad = CodeValidator.validate_code(
        "#include <iostream>\nint main() { std::cout << 123; \n",
        language="cpp",
    )
    assert val_cpp_bad.valid is False
    assert any("brace" in issue.lower() for issue in val_cpp_bad.issues)


def test_toolchain_manager():
    assert ToolchainManager.normalize_language("Python") == "python"
    assert ToolchainManager.normalize_language("C++") == "cpp"
    assert ToolchainManager.normalize_language("rs") == "rust"

    ext_cpp = ToolchainManager.get_extension_for_language("cpp")
    assert ext_cpp == ".cpp"

    ext_java = ToolchainManager.get_extension_for_language("java")
    assert ext_java == ".java"

    ext_rust = ToolchainManager.get_extension_for_language("rust")
    assert ext_rust == ".rs"
