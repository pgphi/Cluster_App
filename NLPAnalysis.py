import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim_models
from wordcloud import WordCloud



def topic_modeling(word_tokens):
    """
    Function for Topic Modeling. Takes word_tokens list <list> as Input and Outputs:
            - the model <class 'gensim.models.ldamulticore.LdaMulticore'>,
            - an analysis <class 'pyLDAvis._prepare.PreparedData'>,
            - a wordcloud <class 'wordcloud.wordcloud.WordCloud'>.

    General Information about Topic Models:
    A type of statistical language models used for uncovering hidden structure in a collection of texts.
    In a practical and more intuitively, you can think of it as a task of:

        - Dimensionality Reduction:
            where rather than representing a text T in its feature space as {Word_i: count(Word_i, T) for Word_i in
            Vocabulary}, you can represent it in a topic space as {Topic_i: Weight(Topic_i, T) for Topic_i in Topics}

        - Unsupervised Learning:
            where it can be compared to clustering, as in the case of clustering, the number of topics, like the number
            of clusters, is an output parameter. By doing topic modeling, we build clusters of words rather than
            clusters of texts. A text is thus a mixture of all the topics, each having a specific weight

        - Tagging:
            abstract “topics” that occur in a collection of documents that best represents the information in them.
    """

    # Create Dictionary
    id2word = corpora.Dictionary([word_tokens])

    # Create Corpus
    texts = [word_tokens]

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # number of topics
    num_topics = 3

    # Build LDA model Pipeline
    model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)

    # LDA Model analyzes
    model_analysis = pyLDAvis.gensim_models.prepare(model, corpus, id2word)
    model_analysis_as_html = pyLDAvis.save_html(model_analysis, 'clusters.html')  # Save as .html file

    # Create wordcloud
    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: (192, 150, 78),
                      prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)
    topic_words = dict(topics[0][1])
    cloud = cloud.generate_from_frequencies(topic_words, max_font_size=300)

    return [model,
            model_analysis,
            cloud,
            model_analysis_as_html
            ]