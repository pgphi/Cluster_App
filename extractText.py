def extract_text(txt_file):

    """Takes UTF-8 .txt file and returns content as string list"""

    with open("/Users/philipp/Documents/PycharmProjects/Topic_Modeling/Text_Files/" + txt_file) as f:
        lines = f.readlines()

    return lines