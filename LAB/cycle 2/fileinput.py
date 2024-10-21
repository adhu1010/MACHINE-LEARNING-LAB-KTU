import string
from collections import Counter


def get_text_from_user():
    print("Please enter your text. Press Enter twice to finish:")
    text_lines = []
    while True:
        line = input()
        if line == "":  
            break
        text_lines.append(line)
    text = "\n".join(text_lines)   
    with open('user_input.txt', 'w') as file:
        file.write(text)
    return 'user_input.txt'

def read_file(file_path):   
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def process_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def get_most_frequent_words(text, num_words=10):
    words = text.split()
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(num_words)
    return most_common_words

def main():
    file_path = get_text_from_user()
    text = read_file(file_path)
    processed_text = process_text(text)
    most_frequent_words = get_most_frequent_words(processed_text)
    print("\nMost Frequent Words:")
    for count in most_frequent_words:
        co=count
        print(co)
        break
    for word, count in most_frequent_words:
        if count==co:
            print(f"{word}: {count}")
if __name__ == "__main__":
    main()
