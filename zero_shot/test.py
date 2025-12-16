def create_options_dict(answer, answer_str, wrong_options):
    """
    Combine correct and wrong options into a dictionary with letters as keys.
    
    Args:
        answer: The correct answer letter (e.g., 'A', 'B', 'C', 'D')
        answer_str: The correct answer string content
        wrong_options: List of wrong answer strings
    
    Returns:
        dict: Dictionary with letter keys and string values
    
    Example:
        >>> result = create_options_dict('B', 'Paris', ['London', 'Berlin', 'Madrid'])
        >>> result
        {'A': 'London', 'B': 'Paris', 'C': 'Berlin', 'D': 'Madrid'}
    """
    # Generate enough letters for all options
    total_options = len(wrong_options) + 1  # +1 for correct answer
    letters = [chr(65 + i) for i in range(total_options)]  # A, B, C, D, ...
    
    # Create dictionary with correct answer at the specified letter
    options_dict = {answer: answer_str}
    
    # Add wrong options to remaining letters (excluding the correct answer letter)
    available_letters = [l for l in letters if l != answer]
    
    for i, wrong_str in enumerate(wrong_options):
        options_dict[available_letters[i]] = wrong_str
    
    return options_dict


# Example usage
if __name__ == "__main__":
    result = create_options_dict('B', 'Paris', ['London', 'Berlin', 'Madrid'])
    print(result)
    print(f"\nCorrect answer: B -> {result['B']}")
