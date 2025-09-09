

import random
import math
import pickle
import os
from collections import defaultdict, Counter

class PureTextGenerator:
    def __init__(self, n_gram=3):
        self.n_gram = n_gram
        self.model = defaultdict(list)
        self.vocab = set()
        self.training_data = {}
        self.topic_models = {}
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        # Keep only alphanumeric characters, spaces, and basic punctuation
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-')
        text = ''.join(c for c in text if c in allowed_chars)
        return text.lower()
    
    def tokenize(self, text):
        """Simple tokenization"""
        # Split into words and add sentence markers
        words = text.split()
        return ['<START>'] * (self.n_gram - 1) + words + ['<END>']
    
    def build_ngram_model(self, text):
        """Build n-gram model from text"""
        model = defaultdict(list)
        tokens = self.tokenize(text)
        
        for i in range(len(tokens) - self.n_gram + 1):
            context = tuple(tokens[i:i + self.n_gram - 1])
            next_word = tokens[i + self.n_gram - 1]
            model[context].append(next_word)
            
        return model
    
    def train_on_topic(self, topic, texts):
        """Train model on specific topic"""
        print(f"Training on topic: {topic}")
        combined_text = ' '.join(texts)
        processed_text = self.preprocess_text(combined_text)
        
        # Build topic-specific model
        self.topic_models[topic] = self.build_ngram_model(processed_text)
        
        # Update vocabulary
        tokens = self.tokenize(processed_text)
        self.vocab.update(tokens)
        
        print(f"Added {len(tokens)} tokens for topic {topic}")
    
    def train(self, training_data):
        """Train the model on all topics"""
        print("Starting training...")
        self.training_data = training_data
        
        for topic, texts in training_data.items():
            self.train_on_topic(topic, texts)
        
        # Build combined model
        all_texts = []
        for texts in training_data.values():
            all_texts.extend(texts)
        
        combined_text = ' '.join(all_texts)
        processed_text = self.preprocess_text(combined_text)
        self.model = self.build_ngram_model(processed_text)
        
        print(f"Training complete! Vocabulary size: {len(self.vocab)}")
        print(f"Total n-grams: {sum(len(v) for v in self.model.values())}")
    
    def generate_text(self, length=100, topic=None, seed=None, temperature=1.0):
        """Generate text using the trained model"""
        # Choose model based on topic
        if topic and topic in self.topic_models:
            model = self.topic_models[topic]
        else:
            model = self.model
        
        if not model:
            return "Model not trained yet!"
        
        # Initialize context
        if seed:
            seed_tokens = self.tokenize(self.preprocess_text(seed))
            if len(seed_tokens) >= self.n_gram - 1:
                context = tuple(seed_tokens[-(self.n_gram - 1):])
            else:
                context = tuple(['<START>'] * (self.n_gram - 1))
        else:
            context = tuple(['<START>'] * (self.n_gram - 1))
        
        generated = []
        
        for _ in range(length):
            if context in model:
                candidates = model[context]
                
                # Apply temperature-based sampling
                if temperature != 1.0:
                    # Count frequencies
                    word_counts = Counter(candidates)
                    words = list(word_counts.keys())
                    counts = list(word_counts.values())
                    
                    # Apply temperature
                    if temperature > 0:
                        probs = [math.pow(count, 1.0/temperature) for count in counts]
                        total = sum(probs)
                        probs = [p/total for p in probs]
                        
                        # Weighted random selection
                        rand = random.random()
                        cumulative = 0
                        next_word = words[0]
                        for i, prob in enumerate(probs):
                            cumulative += prob
                            if rand <= cumulative:
                                next_word = words[i]
                                break
                    else:
                        next_word = max(word_counts, key=word_counts.get)
                else:
                    next_word = random.choice(candidates)
                
                if next_word == '<END>':
                    break
                
                generated.append(next_word)
                # Update context
                context = context[1:] + (next_word,)
            else:
                # If context not found, try shorter context or random word
                if len(context) > 1:
                    context = context[1:]
                else:
                    # Pick random word from vocabulary
                    vocab_words = [w for w in self.vocab if w not in ['<START>', '<END>']]
                    if vocab_words:
                        next_word = random.choice(vocab_words)
                        generated.append(next_word)
                        context = context[1:] + (next_word,)
                    else:
                        break
        
        return ' '.join(generated)
    
    def save_model(self, filename):
        """Save trained model to file"""
        model_data = {
            'n_gram': self.n_gram,
            'model': dict(self.model),
            'topic_models': {k: dict(v) for k, v in self.topic_models.items()},
            'vocab': list(self.vocab),
            'training_data': self.training_data
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load trained model from file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.n_gram = model_data['n_gram']
        self.model = defaultdict(list, model_data['model'])
        self.topic_models = {k: defaultdict(list, v) for k, v in model_data['topic_models'].items()}
        self.vocab = set(model_data['vocab'])
        self.training_data = model_data['training_data']
        
        print(f"Model loaded from {filename}")

class TextAnalyzer:
    @staticmethod
    def analyze_text_quality(text):
        """Analyze generated text quality"""
        if not text:
            return {"words": 0, "sentences": 0, "avg_word_length": 0}
        
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        return {
            "words": len(words),
            "sentences": len(sentences),
            "avg_word_length": avg_word_length,
            "unique_words": len(set(words)),
            "repetition_rate": 1 - (len(set(words)) / len(words)) if words else 0
        }
    
    @staticmethod
    def topic_relevance(text, topic):
        """Check topic relevance"""
        topic_keywords = {
            'technology': ['technology', 'computer', 'digital', 'software', 'artificial', 'machine', 'algorithm', 'data', 'internet', 'cyber', 'programming', 'code', 'system'],
            'science': ['science', 'research', 'experiment', 'discovery', 'theory', 'universe', 'climate', 'genetic', 'biology', 'physics', 'chemistry', 'study', 'analysis'],
            'literature': ['literature', 'poetry', 'story', 'author', 'book', 'novel', 'writing', 'narrative', 'character', 'plot', 'poem', 'verse', 'prose']
        }
        
        if topic not in topic_keywords:
            return 0
        
        text_lower = text.lower()
        matches = sum(1 for keyword in topic_keywords[topic] if keyword in text_lower)
        return matches / len(topic_keywords[topic])

def create_sample_data():
    """Create sample training data"""
    return {
        'technology': [
            "Artificial intelligence is revolutionizing how we work and live. Machine learning algorithms can now perform complex tasks that once required human intelligence. Deep learning networks are advancing computer vision and natural language processing. The future holds exciting possibilities for quantum computing and neural interfaces.",
            
            "The internet has connected billions of people worldwide. Social media platforms have transformed communication and information sharing. Cloud computing makes powerful resources accessible to everyone. Mobile technology puts supercomputers in our pockets. Cybersecurity protects our digital infrastructure from threats.",
            
            "Programming languages enable developers to create software applications. Code optimization improves system performance and efficiency. Database management systems store and retrieve information quickly. Web development frameworks simplify building interactive websites. Software engineering practices ensure reliable and maintainable code.",
            
            "Digital transformation is changing business operations across industries. Automation streamlines repetitive tasks and increases productivity. Data analytics provides insights for better decision making. Internet of things devices collect real-time information. Blockchain technology offers secure and transparent transactions."
        ],
        
        'science': [
            "Scientific research drives human understanding of the natural world. Experiments test hypotheses and validate theories. Peer review ensures research quality and reliability. Collaboration accelerates discovery and innovation. The scientific method provides a framework for investigating phenomena.",
            
            "Physics explores the fundamental laws governing matter and energy. Chemistry studies molecular interactions and reactions. Biology investigates living organisms and their processes. Mathematics provides tools for modeling and analysis. Astronomy examines celestial objects and cosmic phenomena.",
            
            "Climate science studies weather patterns and environmental changes. Genetics research unlocks secrets of heredity and evolution. Neuroscience investigates brain function and consciousness. Medical research develops treatments for diseases. Environmental science addresses conservation and sustainability challenges.",
            
            "Laboratory equipment enables precise measurements and observations. Data collection methods ensure accurate and reliable results. Statistical analysis reveals patterns and relationships. Computer simulations model complex systems. Interdisciplinary collaboration combines expertise from multiple fields."
        ],
        
        'literature': [
            "Literature reflects human experiences and cultural values. Authors use language to create vivid imagery and explore emotions. Narrative techniques engage readers and convey meaning. Literary criticism analyzes themes and interpretations. Classic works continue to influence contemporary writing.",
            
            "Poetry expresses ideas through rhythm, meter, and figurative language. Poets capture beauty, emotion, and human experience in verse. Different poetic forms offer unique structures and styles. Metaphors and symbols add layers of meaning. Spoken word performances bring poetry to life.",
            
            "Storytelling is a fundamental human activity that connects people. Oral traditions preserve cultural knowledge and wisdom. Modern narratives explore contemporary issues and experiences. Character development creates memorable and relatable figures. Plot structure guides readers through engaging journeys.",
            
            "Creative writing workshops help authors develop their craft. Reading widely exposes writers to different styles and techniques. Revision and editing refine and improve written work. Publishing platforms make literature accessible to global audiences. Book clubs and literary societies foster community discussion."
        ]
    }

def interactive_demo():
    """Interactive demonstration of the text generator"""
    print("=" * 60)
    print("CODTECH INTERNSHIP - TASK 4")
    print("GENERATIVE TEXT MODEL DEMO")
    print("=" * 60)
    
    # Initialize generator
    generator = PureTextGenerator(n_gram=3)
    
    # Load or create training data
    print("\nInitializing training data...")
    training_data = create_sample_data()
    
    # Train the model
    generator.train(training_data)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    # Interactive loop
    while True:
        print("\n" + "-" * 40)
        print("OPTIONS:")
        print("1. Generate text on specific topic")
        print("2. Generate text with custom seed")
        print("3. Compare different temperatures")
        print("4. Analyze text quality")
        print("5. Show model statistics")
        print("6. Save model")
        print("7. Exit")
        print("-" * 40)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            print("\nAvailable topics: technology, science, literature")
            topic = input("Enter topic: ").strip().lower()
            
            if topic in training_data:
                length = int(input("Enter length (default 50): ") or 50)
                temperature = float(input("Enter temperature 0.1-2.0 (default 1.0): ") or 1.0)
                
                print(f"\nGenerating {length} words on '{topic}'...")
                text = generator.generate_text(length=length, topic=topic, temperature=temperature)
                print(f"\nGenerated Text:\n{text}")
                
                # Analyze quality
                analysis = TextAnalyzer.analyze_text_quality(text)
                relevance = TextAnalyzer.topic_relevance(text, topic)
                
                print(f"\nQuality Analysis:")
                print(f"Words: {analysis['words']}")
                print(f"Unique words: {analysis['unique_words']}")
                print(f"Average word length: {analysis['avg_word_length']:.2f}")
                print(f"Topic relevance: {relevance:.2f}")
            else:
                print("Invalid topic!")
        
        elif choice == '2':
            seed = input("Enter seed text: ").strip()
            length = int(input("Enter length (default 50): ") or 50)
            temperature = float(input("Enter temperature 0.1-2.0 (default 1.0): ") or 1.0)
            
            print(f"\nGenerating text from seed: '{seed}'...")
            text = generator.generate_text(length=length, seed=seed, temperature=temperature)
            print(f"\nGenerated Text:\n{text}")
        
        elif choice == '3':
            topic = input("Enter topic (technology/science/literature): ").strip().lower()
            seed = input("Enter seed (optional): ").strip() or None
            
            temperatures = [0.5, 1.0, 1.5]
            
            for temp in temperatures:
                print(f"\n--- Temperature {temp} ---")
                text = generator.generate_text(length=30, topic=topic, seed=seed, temperature=temp)
                print(text)
        
        elif choice == '4':
            text = input("Enter text to analyze: ").strip()
            
            if text:
                analysis = TextAnalyzer.analyze_text_quality(text)
                print(f"\nText Analysis:")
                print(f"Words: {analysis['words']}")
                print(f"Sentences: {analysis['sentences']}")
                print(f"Unique words: {analysis['unique_words']}")
                print(f"Average word length: {analysis['avg_word_length']:.2f}")
                print(f"Repetition rate: {analysis['repetition_rate']:.2f}")
                
                # Check relevance to all topics
                for topic in training_data.keys():
                    relevance = TextAnalyzer.topic_relevance(text, topic)
                    print(f"Relevance to {topic}: {relevance:.2f}")
        
        elif choice == '5':
            print(f"\nModel Statistics:")
            print(f"N-gram size: {generator.n_gram}")
            print(f"Vocabulary size: {len(generator.vocab)}")
            print(f"Total n-grams: {sum(len(v) for v in generator.model.values())}")
            print(f"Topics trained: {list(generator.topic_models.keys())}")
            
            for topic in generator.topic_models:
                topic_ngrams = sum(len(v) for v in generator.topic_models[topic].values())
                print(f"N-grams for {topic}: {topic_ngrams}")
        
        elif choice == '6':
            filename = input("Enter filename (default: text_model.pkl): ").strip() or "text_model.pkl"
            generator.save_model(filename)
        
        elif choice == '7':
            print("Thank you for using the Generative Text Model!")
            break
        
        else:
            print("Invalid choice! Please enter 1-7.")

def run_automated_demo():
    """Run automated demonstration"""
    print("=" * 60)
    print("AUTOMATED DEMO - GENERATIVE TEXT MODEL")
    print("=" * 60)
    
    generator = PureTextGenerator(n_gram=3)
    training_data = create_sample_data()
    generator.train(training_data)
    
    print("\n1. TOPIC-BASED GENERATION:")
    print("-" * 30)
    
    for topic in ['technology', 'science', 'literature']:
        print(f"\n{topic.upper()}:")
        text = generator.generate_text(length=40, topic=topic, temperature=0.8)
        print(text)
        
        analysis = TextAnalyzer.analyze_text_quality(text)
        relevance = TextAnalyzer.topic_relevance(text, topic)
        print(f"Quality: {analysis['words']} words, {analysis['unique_words']} unique")
        print(f"Relevance: {relevance:.2f}")
    
    print("\n\n2. TEMPERATURE COMPARISON:")
    print("-" * 30)
    
    seed = "artificial intelligence"
    for temp in [0.5, 1.0, 1.5]:
        print(f"\nTemperature {temp}:")
        text = generator.generate_text(length=25, seed=seed, temperature=temp)
        print(text)
    
    print("\n\n3. SEED-BASED GENERATION:")
    print("-" * 30)
    
    seeds = ["machine learning", "scientific research", "creative writing"]
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        text = generator.generate_text(length=30, seed=seed, temperature=1.0)
        print(text)

if __name__ == "__main__":
    print("CODTECH INTERNSHIP - TASK 4")
    print("Generative Text Model (Pure Python)")
    print("=" * 40)
    
    mode = input("Choose mode:\n1. Interactive Demo\n2. Automated Demo\nEnter choice (1/2): ").strip()
    
    if mode == '1':
        interactive_demo()
    elif mode == '2':
        run_automated_demo()
    else:
        print("Running automated demo...")
        run_automated_demo()
