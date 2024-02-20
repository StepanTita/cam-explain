# Explanation Benchmark

**********************************************************************************************************************
Explanation:

- Train the model on some dataset (e.g. Fake News Kaggle Competition, or Covid Fake News Dataset)
- Let's embed the whole fact-checking dataset into the same space (not the embeddings, but the centroid of the
  embeddings).
- Let's try to predict the news to be fake or true.
- After we've predicted let's use the embedding centroid and extract k nearest neighbors from the knowledge base.
- Let's see how well this nearest neighbors match the original fact-checking articles (this we will measure by
  max/average cosine/euclidian similarity of the neighbors embedding with the original fact-checking articles,
  and by the number of exact matches, e.g. recall and precision).

Notes: we should actually compare this with S-BERT. For now we just use Mean of BERT embeddings as a centroid.
We should also think how we can deal with the fact that to use concept space similarity we need to use `N` knowledge
bases - one for each label. Since otherwise vectors that fall in different spaces will be compared (since we force
them to be orthogonal only with those from the different concept spaces).

**********************************************************************************************************************

tp / (tp + fp) = precision (how well we identify true explanation)
tp / (tp + fn) = recall (how well we distinguish true explanation from false explanation)

| Model        | Train Dataset    | Epochs | Train Params | Mean Cosine Similarity | Max Cosine Similarity | Mean Euclidean Distance | Max Euclidean Distance | precision | recall |
|--------------|------------------|--------|--------------|------------------------|-----------------------|-------------------------|------------------------|-----------|--------|
| BERT ✅       | Covid Fake       | 50     | 1538         | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |
| BERT ✅       | Fake News Kaggle | 50     | 1538         | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |
| Space-BERT ✅ | Covid Fake       | 50     | 4622 (3)     | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |
| Space-BERT ✅ | Covid Fake       | 50     | 98562 (64)   | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |
| Space-BERT ✅ | Covid Fake       | 50     | 197122 (128) | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |
| Space-BERT ✅ | Fake News Kaggle | 50     | 4622 (3)     | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |
| Space-BERT ✅ | Fake News Kaggle | 50     | 98562 (64)   | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |
| Space-BERT ✅ | Fake News Kaggle | 50     | 197122 (128) | 0.                     | 0.                    | 0.                      | 0.                     | 0.        | 0.     |

## GoEmotions

### Basic Metrics

| Model            | # Latent Dimensions | loss   | accuracy | cs accuracy | f1     | cs f1  | precision | recall |
|------------------|---------------------|--------|----------|-------------|--------|--------|-----------|--------|
| DistilBERT       | N/A                 | 0.1307 | 0.0707   | N/A         | 0.0598 | N/A    | 0.3291    | 0.0388 |
| Space-DistilBERT | 3                   | 0.1925 | 0.0910   | 0.2749      | 0.0812 | 0.0560 | 0.3807    | 0.0557 |
| Space-DistilBERT | 64                  | 0.     | 0.       | 0.          | 0.     | 0.     | 0.        | 0.     |
| Space-DistilBERT | 128                 | 0.     | 0.       | 0.          | 0.     | 0.     | 0.        | 0.     |

### Explanations

| Model            | # Latent Dims. | Metric | Jaccard | Mean Cosine | Min Cosine | Mean Euclid | Max Euclid | Accuracy | F1 Score | Precision | Recall |
|------------------|----------------|--------|---------|-------------|------------|-------------|------------|----------|----------|-----------|--------|
| DistilBERT       | N/A            | TOP 1  | 0.0817  | 0.9782      | 0.9782     | 3.1603      | 3.1602     | 0.0870   | 0.0457   | 0.1232    | 0.0799 |
|                  | N/A            | TOP 3  | 0.0752  | 0.9774      | 0.9766     | 3.2252      | 3.2796     | 0.1154   | 0.0872   | 0.1116    | 0.2054 |
|                  | N/A            | TOP 5  | 0.0705  | 0.9768      | 0.9758     | 3.2649      | 3.3371     | 0.1188   | 0.1009   | 0.0956    | 0.3069 |
| Space-DistilBERT | 3              | TOP 1  | 0.1519  | 0.8230      | 0.8230     | 3.7735      | 3.7735     | 0.1599   | 0.0967   | 0.1786    | 0.0977 |
|                  |                | TOP 3  | 0.1229  | 0.7967      | 0.7704     | 4.0580      | 4.2929     | 0.1871   | 0.1528   | 0.1393    | 0.2636 |
|                  |                | TOP 5  | 0.1098  | 0.7789      | 0.7394     | 4.2341      | 4.5597     | 0.1833   | 0.1616   | 0.1178    | 0.4037 |
| Space-DistilBERT | 64             | TOP 1  | 0.      | 0.          | 0.         | .           | .          | 0.       | 0.       | 0.        | 0.     |
|                  |                | TOP 3  | 0.      | 0.          | 0.         | .           | .          | 0.       | 0.       | 0.        | 0.     |
|                  |                | TOP 5  | 0.      | 0.          | 0.         | .           | .          | 0.       | 0.       | 0.        | 0.     |
| Space-DistilBERT | 128            | TOP 1  | 0.      | 0.          | 0.         | .           | .          | 0.       | 0.       | 0.        | 0.     |
|                  |                | TOP 3  | 0.      | 0.          | 0.         | .           | .          | 0.       | 0.       | 0.        | 0.     |
|                  |                | TOP 5  | 0.      | 0.          | 0.         | .           | .          | 0.       | 0.       | 0.        | 0.     |
