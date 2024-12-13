class TriangularFuzzyNumber:
    def __init__(self, left, peak, right):
        """
        Initialize a Triangular Fuzzy Number.

        Parameters:
        - left (float): The lower limit.
        - peak (float): The most probable value.
        - right (float): The upper limit.
        """
        self.left = left
        self.peak = peak
        self.right = right

    def __add__(self, other):
        """
        Add two Triangular Fuzzy Numbers.

        Parameters:
        - other (TriangularFuzzyNumber): The fuzzy number to add.

        Returns:
        - TriangularFuzzyNumber: The result of addition.
        """
        return TriangularFuzzyNumber(
            self.left + other.left,
            self.peak + other.peak,
            self.right + other.right
        )

    def __sub__(self, other):
        """
        Subtract two Triangular Fuzzy Numbers.

        Parameters:
        - other (TriangularFuzzyNumber): The fuzzy number to subtract.

        Returns:
        - TriangularFuzzyNumber: The result of subtraction.
        """
        return TriangularFuzzyNumber(
            self.left - other.right,
            self.peak - other.peak,
            self.right - other.left
        )

    def defuzzify(self):
        """
        Defuzzify the Triangular Fuzzy Number using the Centroid Method.

        Returns:
        - float: The crisp value.
        """
        return (self.left + self.peak + self.right) / 3

    def __str__(self):
        return f"TriangularFuzzyNumber(left={self.left}, peak={self.peak}, right={self.right})"

    def __repr__(self):
        return self.__str__()
