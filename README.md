# **_Sentiment Analysis on Synthetic Social Media_**


_This project compares **Natural Language Processing (NLP)** techniques with **Recurrent Neural Networks (RNN)** and **Gated Recurrent Units (GRU)** for sentiment analysis on synthetic social media data. It aims to evaluate the effectiveness of these computational approaches in identifying and classifying sentiments, highlighting the superiority of NLP models in terms of precision and performance._

# **_Aim_**
_To analyze and compare the performance of sentiment analysis models, focusing on NLP, RNN, and GRU architectures, and provide insights into the most effective approach for processing synthetic social media data._

# _**Features**_

**_1. Dataset :_** _A synthetic social media dataset curated to represent diverse sentiments and emulate real-world scenarios._

**_2. Preprocessing Steps :_** 

 - _Tokenization_
 - _Stop word removal_
 - _One-hot encoding of labels_
 - _Addressing class imbalance for unbiased results_

_**3. Algorithms :**_

- **_NLP Model (Bidirectional LSTM) :-_** _Captures context from both past and future sequences._
- _**RNN (SimpleRNN) :-**_ _Processes sequential data using a simpler recurrent architecture._
- _**GRU :-**_ _Balances performance and efficiency by reducing parameters compared to LSTMs._

_**4. Evaluation Metrics :**_

- _Accuracy_
- _Precision_
- _Recall_
- _F1-score_

_**5. Comparison Highlights :**_ _NLP outperformed RNN and GRU in overall sentiment classification accuracy and contextual understanding._
# _**Methodology**_

_**1. Dataset Preparation :**_

- _**Source :-**_ _Synthetic social media dataset._
- _**Preprocessing :-**_ _Applied tokenization, normalization, stop word removal, and one-hot encoding to prepare data for training._

_**2. Model Architectures :**_

- _**NLP (Bidirectional LSTM) :-**_ _Utilized for its capability to capture bi-directional dependencies in textual data._
- _**RNN (SimpleRNN) :-**_ _Employed for comparison due to its simplicity in processing sequential data._
- _**GRU :-**_ _Chosen for its computational efficiency and ability to handle vanishing gradient issues._

_**3. Training and Testing :**_

- _Models trained using categorical cross-entropy loss and the Adam optimizer._
- _Evaluation on a test set using performance metrics: accuracy, precision, recall, and F1-score._

_**4. Statistical Analysis :**_

- _Statistical significance of performance differences was tested using IBM SPSS to ensure robustness in comparative results._

# _**Results**_
_**Performance Metrics :**_

<img width="728" alt="1234" src="https://github.com/user-attachments/assets/8d0c1b4c-13b6-4896-b967-40351c5cc381">


- _**NLP Model** demonstrated superior performance in precision and recall, making it the most effective model for sentiment analysis._
- _GRU outperformed RNN in accuracy and computational efficiency, offering a middle ground between simplicity and effectiveness._

# _**Advantages**_

- _**NLP Model :-** Exceptional at understanding complex sentiment patterns due to its bidirectional architecture._
- _**GRU :-** Efficient and lightweight, suitable for large datasets._
- _**RNN :-** Provides a baseline for sequential data processing._

# _**Limitations**_

- _**Synthetic Data :-** Real-world social media data could introduce challenges like noise and sarcasm._
- _**Scalability :-** Larger datasets may require advanced computational resources._
- _**Class Imbalance :-** Despite addressing it, further refinements could improve model generalizability._

# _**Future Scope**_

- _Explore real-world datasets to validate findings._
- _Incorporate attention mechanisms for improved performance._
- _Optimize GRU architecture for further efficiency gains._

# _**How to Run the Code**_

_**Prerequisites :**_

- _**Programming Language :-** Python._
- _**Libraries :-** TensorFlow, Keras, NumPy, Pandas, Scikit-learn._
- _**Statistical Analysis Tool :-** IBM SPSS_

_**Steps :**_

_**1. Clone the repository :-**_

- _`git clone https://github.com/muralioleti/sentiment-analysis-comparison`_

_**2. Install dependencies :-**_

- _`pip install -r requirements.txt`_


_**3. Run the preprocessing script :-**_

- _`python preprocess_data.py`_

_**4. Train the models :-**_

- _`python train_models.py`_

_**5. Evaluate results :-**_

- _`python evaluate_models.py`_

# _**Conclusion**_

_The study highlights that **NLP model** outperform **RNN** and **GRU** in sentiment analysis on synthetic social media data. These findings emphasize the potential of advanced NLP techniques in capturing nuanced emotional patterns, paving the way for more robust real-world applications._
