import random
from collections import defaultdict

class MarkovTextGenerator:
    def __init__(self, order=2):
        """
        Initialize the Markov chain text generator.
        
        Args:
            order: Number of words to use as state (default: 2)
        """
        self.order = order
        self.transitions = defaultdict(list)
        self.starts = []
    
    def train(self, text):
        """
        Train the generator on input text.
        
        Args:
            text: String of text to train on
        """
        words = text.split()
        
        # Store possible starting states
        for i in range(len(words) - self.order):
            state = tuple(words[i:i + self.order])
            self.starts.append(state)
            
            # Add next word to transitions
            next_word = words[i + self.order]
            self.transitions[state].append(next_word)
    
    def generate(self, length=50):
        """
        Generate text using the trained Markov chain.
        
        Args:
            length: Number of words to generate
            
        Returns:
            Generated text as string
        """
        if not self.starts:
            return "No training data available."
        
        # Start with random state
        current_state = random.choice(self.starts)
        output = list(current_state)
        
        # Generate words
        for _ in range(length - self.order):
            if current_state not in self.transitions:
                break
            
            next_word = random.choice(self.transitions[current_state])
            output.append(next_word)
            
            # Update state by shifting window
            current_state = tuple(output[-self.order:])
        
        return ' '.join(output)


# Example usage
if __name__ == "__main__":
    # Sample training text
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    The lazy dog sleeps under the tree.
    The brown fox is very quick and clever.
    The clever fox hunts in the forest.
    """
    
    # Create and train generator
    generator = MarkovTextGenerator(order=2)
    generator.train(sample_text)
    
    # Generate text
    print("Generated Text:")
    for i in range(5):
        print(f"{i+1}. {generator.generate(length=20)}")