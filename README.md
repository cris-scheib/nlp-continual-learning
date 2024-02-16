# Continuous Learning Applied to the Development of a Chatbot Based on Sequence-To-Sequence Architecture

The evolution of chatbot tools offers significant opportunities, enabling interactive interactions beneficial to businesses and customers. 
Research outside of Natural Language Processing helps improve the accuracy of these tools, but the volatility of the language presents major challenges, requiring ongoing training to avoid obsolescence. 
Continuous learning is a promising approach to improve the adaptation of models to dynamic environments and enable them to evolve with changing demands. 
This work aims to investigate the benefits and challenges of continuous learning techniques in the development of a chatbot based on the Sequence-to-Sequence architecture. 
This makes it possible to identify the possibility of implementing adaptive and progressive training through feedback from the model based on user interactions. 
As a result, continuous learning provides resilience, allowing the assimilation of knowledge without compromising previously obtained information. 
Although the Sequence-to-Sequence architecture shows promising results in abstracting sequential data, the need for more intensive training is crucial to ensure high model quality. 
This implies that versatile and flexible models require robust hardware, high-quality data, and extended training periods.

----------------------------

### Tools
The [Keras Framework](https://keras.io/) was used to create the training architecture. 
Tools such as [Pandas](https://pandas.pydata.org/), [Numpy](https://numpy.org/) and [Matplotlib](https://matplotlib.org/), among other tools, were also used.

### Structure

The Sequence-to-sequence (Seq2Seq) model is an architecture composed of three main components: the Encoder, the attention layer and the Decoder. The Encoder converts the input sequence into a one-dimensional vector, which is then processed by the attention layer to allow the Decoder to flexibly utilize relevant parts of the input. The Decoder generates the output sequence based on the information provided by the Encoder and the attention layer.

The Encoder consists of an Embedding layer followed by a GRU layer, which processes the input sequence and produces a context vector summarizing all relevant information. This context vector is then sent to the attention layer, which forces the network to review the input sequence to capture additional information.

The Decoder, similar to the Encoder, also has an Embedding layer, a GRU layer and a dense layer. It receives the context vector from the attention layer as the initial state and synthesizes the relevant information to generate the output sequence. During output generation, the Decoder produces a sparse vector that describes the probability of each token in the vocabulary, choosing the most likely token at each step.

During inference, the Seq2Seq model is fed the source sequence through the Encoder, obtains the context vector, and synthesizes it through the attention layer. This vector is then passed to the Decoder, which generates the output sequence token by token, with each token being used as subsequent input.

The Seq2Seq architecture has a large number of training variables, totaling 59 million trainable parameters, distributed between the Encoder, the Decoder and the attention layer. These variables are adjusted during training to improve the model's performance in its task.



![image](https://github.com/cris-scheib/nlp-continual-learning/assets/61483993/5416d5f7-2909-4698-b4bd-cf41bed57617)

