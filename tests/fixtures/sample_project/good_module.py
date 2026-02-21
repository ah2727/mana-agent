"""Sample module docstring."""


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class Greeter:
    """Simple greeter."""

    def __init__(self, name: str) -> None:
        """Initialize name."""
        self.name = name

    def greet(self) -> str:
        """Produce greeting."""
        return f"Hello {self.name}"
