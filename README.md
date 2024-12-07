
### Semantic_search

I built a semantic search engine using OpenAI API that takes user input text, which is transformed in vector embedding, and then returns first k texts from MongoDB Atlas, where a bunch of texts and their vectors are stored.
These texts are reviews from dataset: https://raw.githubusercontent.com/keitazoumana/Experimentation-Data/main/Musical_instruments_reviews.csv
For now I checked the results and:
    When I input a short positive sentiment("I like this" or " I dont like it") there are some bad returns, but when using longer similar texts that are existing in the stored text, there is a visible difference in their similarity value. 