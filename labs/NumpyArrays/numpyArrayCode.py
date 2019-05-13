
import numpy as np
import time;
# there are 2000 words in the corpus
alpha = np.full (2000, .1)
 
# there are 100 topics
beta = np.full (100, .1)
# this gets us the probabilty of each word happening in each of the 100 topics
wordsInTopic = np.random.dirichlet (alpha, 100)
# produced [doc, topic, word] gives us the number of times that the given word was
# produced by the given topic in the given doc
produced = np.zeros ((50, 100, 2000))
 
# generate each doc
for doc in range (0, 50):
        #
        # get the topic probabilities for this doc
        topicsInDoc = np.random.dirichlet (beta)
        #
        # assign each of the 2000 words in this doc to a topic
        wordsToTopic = np.random.multinomial (2000, topicsInDoc)
        #
        # and generate each of the 2000 words
        for topic in range (0, 100):
                produced[doc, topic] = np.random.multinomial (wordsToTopic[topic], wordsInTopic[topic])

