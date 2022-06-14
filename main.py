from extractText import extract_text
from preprocess import preprocess_data
from NLPAnalysis import topic_modeling

if __name__ == '__main__':
    # Get file
    print("type in your file name i.e. 'Example.txt' (make sure it is UTF-8 encoded!):")
    file = input()

    # Extract file content to list <str>
    txt_file = extract_text(file)

    # Preprocess file content and create tokens
    txt_tokens = preprocess_data(txt_file)[1]
    # print(txt_tokens)

    # Create topic clusters and output a html topic model
    topic_model = topic_modeling(txt_tokens)[3]
