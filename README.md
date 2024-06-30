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

| Model            | # Latent Dimensions | loss    | accuracy | cs accuracy | f1     | cs f1  | precision | recall |
|------------------|---------------------|---------|----------|-------------|--------|--------|-----------|--------|
| DistilBERT       | N/A                 | 0.1307  | 0.0707   | N/A         | 0.0598 | N/A    | 0.3291    | 0.0388 |
| Space-DistilBERT | 3                   | 0.1925  | 0.0910   | 0.2749      | 0.0812 | 0.0560 | 0.3807    | 0.0557 |
| Space-DistilBERT | 64                  | 0.2394  | 0.1054   | 0.2868      | 0.1065 | 0.0553 | 0.4527    | 0.0717 |
| Space-DistilBERT | 128                 | 0.2691  | 0.1047   | 0.2897      | 0.1055 | 0.0559 | 0.4822    | 0.0704 |
| RoBERTa          | N/A                 | 0.1427  | 0.0169   | N/A         | 0.0014 | N/A    | 0.0479    | 0.0007 |
| Space-RoBERTa    | 3                   | 15.4136 | 0.1024   | 0.1860      | 0.0990 | 0.0730 | 0.3005    | 0.0716 |
| Space-RoBERTa    | 64                  | 46.8452 | 0.1051   | 0.2614      | 0.1370 | 0.0457 | 0.2658    | 0.1002 |
| Space-RoBERTa    | 128                 | 97.5496 | 0.0980   | 0.2386      | 0.1334 | 0.0391 | 0.2470    | 0.0975 |

### Explanations

# TODO: make a separate script to calculate these metrics

| Model            | # Latent Dims. | Metric | Top K | Jaccard    | Mean Cosine | Min Cosine | Mean Euclid | Max Euclid | Accuracy   | F1 Score   | Precision  | Recall     |
|------------------|----------------|--------|-------|------------|-------------|------------|-------------|------------|------------|------------|------------|------------|
| DistilBERT       | N/A            | Euclid | TOP 1 | 0.0817     | 0.9782      | 0.9782     | 3.1603      | 3.1602     | 0.0870     | 0.0457     | 0.1232     | 0.0799     |
|                  | N/A            |        | TOP 3 | 0.0752     | 0.9774      | 0.9766     | 3.2252      | 3.2796     | 0.1154     | 0.0872     | 0.1116     | 0.2054     |
|                  | N/A            |        | TOP 5 | 0.0705     | 0.9768      | 0.9758     | 3.2649      | 3.3371     | 0.1188     | 0.1009     | 0.0956     | 0.3069     |
| Space-DistilBERT | 3              |        | TOP 1 | 0.1519     | 0.8230      | 0.8230     | 3.7735      | 3.7735     | 0.1599     | 0.0967     | **0.1786** | 0.0977     |
|                  |                |        | TOP 3 | 0.1229     | 0.7967      | 0.7704     | 4.0580      | 4.2929     | **0.1871** | 0.1528     | 0.1393     | 0.2636     |
|                  |                |        | TOP 5 | 0.1098     | 0.7789      | 0.7394     | 4.2341      | 4.5597     | 0.1833     | 0.1616     | 0.1178     | 0.4037     |
| Space-DistilBERT | 64             |        | TOP 1 | 0.0416     | 0.7930      | 0.7930     | 9.9450      | 9.9451     | 0.0495     | 0.0276     | 0.0971     | 0.0466     |
|                  |                |        | TOP 3 | 0.0488     | 0.6889      | 0.6148     | 11.6900     | 12.8111    | 0.0782     | 0.0533     | 0.0915     | 0.1474     |
|                  |                |        | TOP 5 | 0.0500     | 0.6414      | 0.5481     | 12.4993     | 13.9659    | 0.0860     | 0.0630     | 0.0762     | 0.2388     |
| Space-DistilBERT | 128            |        | TOP 1 | 0.0374     | 0.8567      | 0.8567     | 11.7750     | 11.7750    | 0.0451     | 0.0256     | 0.0868     | 0.0446     |
|                  |                |        | TOP 3 | 0.0442     | 0.7363      | 0.6452     | 13.8267     | 15.3113    | 0.0711     | 0.0465     | 0.0767     | 0.1375     |
|                  |                |        | TOP 5 | 0.0433     | 0.6819      | 0.5718     | 14.7505     | 16.5501    | 0.0749     | 0.0495     | 0.0640     | 0.2151     |
| RoBERTa          | N/A            |        | TOP 1 | 0.1093     | 0.9957      | 0.9957     | 1.0776      | 1.0776     | 0.1078     | 0.0372     | 0.0534     | 0.0409     |
|                  | N/A            |        | TOP 3 | 0.0753     | 0.9954      | 0.9951     | 1.1125      | 1.1401     | 0.1139     | 0.0549     | 0.0443     | 0.1117     |
|                  | N/A            |        | TOP 5 | 0.0664     | 0.9952      | 0.9949     | 1.1324      | 1.1690     | 0.1115     | 0.0648     | 0.0445     | 0.1845     |
| Space-RoBERTa    | 3              |        | TOP 1 | 0.0561     | 0.9698      | 0.9698     | 1.5013      | 1.5013     | 0.0567     | 0.0282     | 0.0469     | 0.0347     |
|                  |                |        | TOP 3 | 0.0487     | 0.9657      | 0.9619     | 1.6005      | 1.6856     | 0.0746     | 0.0513     | 0.0450     | 0.1057     |
|                  |                |        | TOP 5 | 0.0471     | 0.9630      | 0.9579     | 1.6617      | 1.7725     | 0.0799     | 0.0601     | 0.0440     | 0.1780     |
| Space-RoBERTa    | 64             |        | TOP 1 | 0.1424     | 0.9932      | 0.9932     | 3.7781      | 3.7781     | 0.1361     | 0.0348     | 0.0489     | 0.0406     |
|                  |                |        | TOP 3 | 0.0749     | 0.9910      | 0.9889     | 4.2693      | 4.7223     | 0.1123     | 0.0516     | 0.0460     | 0.1081     |
|                  |                |        | TOP 5 | 0.0604     | 0.9896      | 0.9871     | 4.5777      | 5.1366     | 0.1014     | 0.0596     | 0.0447     | 0.1783     |
| Space-RoBERTa    | 128            |        | TOP 1 | 0.1454     | 0.9975      | 0.9975     | 3.6152      | 3.6153     | 0.1382     | 0.0356     | 0.0472     | 0.0412     |
|                  |                |        | TOP 3 | 0.0738     | 0.9967      | 0.9959     | 4.1352      | 4.6319     | 0.1106     | 0.0509     | 0.0435     | 0.1068     |
|                  |                |        | TOP 5 | 0.0603     | 0.9962      | 0.9953     | 4.4691      | 5.0729     | 0.1014     | 0.0604     | 0.0442     | 0.1800     |
| Model            | # Latent Dims. | Metric | Top K | Jaccard    | Mean Cosine | Min Cosine | Mean Euclid | Max Euclid | Accuracy   | F1 Score   | Precision  | Recall     |
| DistilBERT       | N/A            | Cosine | TOP 1 | 0.0850     | 0.9783      | 0.9783     | 3.1634      | 3.1634     | 0.0905     | 0.0455     | 0.1254     | 0.0805     |
|                  | N/A            |        | TOP 3 | 0.0763     | 0.9774      | 0.9767     | 3.2271      | 3.2846     | 0.1169     | 0.0872     | 0.1104     | 0.2081     |
|                  | N/A            |        | TOP 5 | 0.0713     | 0.9769      | 0.9759     | 3.2664      | 3.3433     | 0.1200     | 0.1025     | 0.0982     | 0.3084     |
| Space-DistilBERT | 3              |        | TOP 1 | **0.1784** | 0.8261      | 0.8261     | 3.8036      | 3.8036     | 0.1832     | 0.0989     | 0.1725     | 0.0973     |
|                  |                |        | TOP 3 | 0.1306     | 0.7992      | 0.7763     | 4.0802      | 4.3468     | 0.1984     | 0.1528     | 0.1381     | 0.2648     |
|                  |                |        | TOP 5 | 0.1141     | 0.7810      | 0.7467     | 4.2517      | 4.6225     | 0.1903     | **0.1623** | 0.1165     | **0.4069** |
| Space-DistilBERT | 64             |        | TOP 1 | 0.0469     | 0.7973      | 0.7973     | 10.0647     | 10.0647    | 0.0557     | 0.0363     | 0.1352     | 0.0517     |
|                  |                |        | TOP 3 | 0.0562     | 0.6934      | 0.6242     | 11.8199     | 13.1033    | 0.0893     | 0.0699     | 0.1017     | 0.1593     |
|                  |                |        | TOP 5 | 0.0586     | 0.6475      | 0.5678     | 12.6531     | 14.4731    | 0.1000     | 0.0808     | 0.0823     | 0.2538     |
| Space-DistilBERT | 128            |        | TOP 1 | 0.0412     | 0.8594      | 0.8594     | 11.8854     | 11.8854    | 0.0494     | 0.0306     | 0.0934     | 0.0473     |
|                  |                |        | TOP 3 | 0.0477     | 0.7394      | 0.6517     | 13.9366     | 15.5507    | 0.0765     | 0.0557     | 0.0818     | 0.1421     |
|                  |                |        | TOP 5 | 0.0474     | 0.6851      | 0.5808     | 14.8602     | 16.9011    | 0.0817     | 0.0627     | 0.0721     | 0.2260     |
| RoBERTa          | N/A            |        | TOP 1 | 0.1185     | 0.9957      | 0.9957     | 1.0825      | 1.0825     | 0.1160     | 0.0362     | 0.0539     | 0.0404     |
|                  | N/A            |        | TOP 3 | 0.0776     | 0.9954      | 0.9952     | 1.1172      | 1.1522     | 0.1174     | 0.0553     | 0.0462     | 0.1143     |
|                  | N/A            |        | TOP 5 | 0.0665     | 0.9953      | 0.9950     | 1.1369      | 1.1863     | 0.1117     | 0.0627     | 0.0441     | 0.1835     |
| Space-RoBERTa    | 3              |        | TOP 1 | 0.0620     | 0.9699      | 0.9699     | 1.5032      | 1.5032     | 0.0621     | 0.0290     | 0.0462     | 0.0350     |
|                  |                |        | TOP 3 | 0.0510     | 0.9657      | 0.9621     | 1.6017      | 1.6889     | 0.0779     | 0.0516     | 0.0447     | 0.1067     |
|                  |                |        | TOP 5 | 0.0486     | 0.9630      | 0.9581     | 1.6628      | 1.7772     | 0.0822     | 0.0605     | 0.0439     | 0.1784     |
| Space-RoBERTa    | 64             |        | TOP 1 | 0.1430     | 0.9932      | 0.9932     | 3.7865      | 3.7865     | 0.1362     | 0.0341     | 0.0459     | 0.0397     |
|                  |                |        | TOP 3 | 0.0756     | 0.9911      | 0.9890     | 4.2769      | 4.7416     | 0.1133     | 0.0512     | 0.0462     | 0.1078     |
|                  |                |        | TOP 5 | 0.0610     | 0.9897      | 0.9871     | 4.5844      | 5.1620     | 0.1022     | 0.0595     | 0.0451     | 0.1785     |
| Space-RoBERTa    | 128            |        | TOP 1 | 0.1445     | 0.9976      | 0.9976     | 3.6521      | 3.6521     | 0.1377     | 0.0338     | 0.0451     | 0.0391     |
|                  |                |        | TOP 3 | 0.0747     | 0.9967      | 0.9960     | 4.1702      | 4.7152     | 0.1120     | 0.0523     | 0.0458     | 0.1092     |
|                  |                |        | TOP 5 | 0.0605     | 0.9963      | 0.9954     | 4.5010      | 5.1834     | 0.1016     | 0.0599     | 0.0443     | 0.1789     |

## Eraser Benchmark

Implement the Eraser benchmark and test it for the eraser datasets.
Test the benchmark for the Hate-Xplain dataset.

### Eraser Benchmark explained:

- Train the model with the dataset, since the documents are too long, we would be sampling only a part of the document
  with annotated rationale.
  ??? How to sample the rationale? (check this with the original paper)
  What if we sample some part of the document that is not relevant and also the part that is relevant and concatenate
  them?
- Take the query and use it as the text_a, and add sampled text as text_b.
- Remove normalization from space model concept space.

### HateXplain Metrics (hard rationale):

- Precision
- Recall
- F1 Score (IOU)
- Accuracy
- Jaccard Index

Calculate those based on the rationale and the predicted rationale (this is the token predictions for the space model).

Use binary cross-entropy with logits here as the metric.

### HateXplain Metrics (soft rationale)

- Exponentially distribute the weights of the true rationale to the adjacent tokens.
- Softmax the updated weights vector.

Recall:

- Multiply hard predicted rationale with true rationale.
- Sum the resulting rationale vectors.
- Create a histogram of the resulting sums (the more in the higher region - the better).

Precision:

- Exponentially distribute the weights of the predicted rationale to the adjacent tokens.
- Multiply predicted rationale by the true hard rationale.
- Sum the resulting rationale vectors.
- Create a histogram of the resulting sums (the more in the higher region - the better).

#### Rationale loss

...

- test on the additional dataset zero-shot when trained with rationale vs without rationale.

| Rationale | Dataset    | Model      | # Latent Dimensions | loss   | acc    | cs acc | f1     | cs f1  | precision | recall | rat. jaccard | rat. acc | rat. f1 | rat. precision | rat. recall |
|-----------|------------|------------|---------------------|--------|--------|--------|--------|--------|-----------|--------|--------------|----------|---------|----------------|-------------|
| Yes       | Hatexplain | BERT       | N/A                 | 0.8120 | 0.7907 | N/A    | 0.7905 | N/A    | 0.7904    | 0.7906 | 0.0247       | 0.0482   | 0.0391  | 0.0247         | 0.1738      |
|           |            | Space-BERT | 3                   | 0.7898 | 0.7898 | 0.     | 0.7896 | 0.     | 0.7895    | 0.7899 | 0.0731       | 0.5672   | 0.1016  | 0.0732         | 0.1731      |
|           |            | Space-BERT | 64                  | 0.6951 | 0.7872 | 0.     | 0.7857 | 0.     | 0.7891    | 0.7852 | 0.0437       | 0.2983   | 0.0658  | 0.0437         | 0.1738      |
|           |            | Space-BERT | 128                 | 0.7237 | 0.7925 | 0.     | 0.7910 | 0.     | 0.7948    | 0.7903 | 0.0247       | 0.0482   | 0.0391  | 0.0247         | 0.1738      |
| No        | Hatexplain | BERT       | N/A                 | 0.4697 | 0.7916 | N/A    | 0.7914 | N/A    | 0.7913    | 0.7916 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 3                   | 0.6589 | 0.7811 | 0.1690 | 0.7806 | 0.1400 | 0.7807    | 0.7807 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 64                  | 0.5710 | 0.7881 | 0.7942 | 0.7868 | 0.7942 | 0.7897    | 0.7862 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 128                 | 0.5963 | 0.7846 | 0.7872 | 0.7828 | 0.7872 | 0.7875    | 0.7822 | N/A          | N/A      | N/A     | N/A            | N/A         |
| Yes       | HSOL       | BERT       | N/A                 | 6.5680 | 0.1782 | N/A    | 0.1962 | N/A    | 0.1747    | 0.4101 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 3                   | 1.8144 | 0.1733 | 0.7490 | 0.1880 | 0.3815 | 0.1564    | 0.4013 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 64                  | 4.4699 | 0.1777 | 0.7742 | 0.2002 | 0.2909 | 0.4218    | 0.4149 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 128                 | 5.2695 | 0.1754 | 0.7740 | 0.1948 | 0.2909 | 0.4033    | 0.4151 | N/A          | N/A      | N/A     | N/A            | N/A         |
| No        | HSOL       | BERT       | N/A                 | 6.4491 | 0.1782 | N/A    | 0.1965 | N/A    | 0.1754    | 0.4103 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 3                   | 2.5876 | 0.1763 | 0.0618 | 0.1899 | 0.0599 | 0.1669    | 0.4033 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 64                  | 4.1446 | 0.1771 | 0.1761 | 0.2011 | 0.1906 | 0.1738    | 0.4144 | N/A          | N/A      | N/A     | N/A            | N/A         |
|           |            | Space-BERT | 128                 | 4.4425 | 0.1762 | 0.1764 | 0.1952 | 0.1900 | 0.1571    | 0.4144 | N/A          | N/A      | N/A     | N/A            | N/A         |

TODO:
* !!! Train only last classifier layer for the space model.
* https://hatebase.org/search_results
* Rational loss to space model


# Latents

| Model          | Dataset    | # Latent Dimensions | loss   | accuracy | cs accuracy | f1     | cs f1  | precision | recall |
|----------------|------------|---------------------|--------|----------|-------------|--------|--------|-----------|--------|
| CAM-DistilBERT | hatespeech | 4                   | 0.7789 | 0.7752   | 0.0508      | 0.2911 | 0.0323 | 0.2584    | 0.3333 |
| CAM-DistilBERT | hatespeech | 8                   | 0.7867 | 0.7752   | 0.0540      | 0.2911 | 0.0444 | 0.2584    | 0.3333 |
| CAM-DistilBERT | hatespeech | 16                  | 0.7381 | 0.7752   | 0.1747      | 0.2911 | 0.0995 | 0.2584    | 0.3333 |
| CAM-DistilBERT | hatespeech | 32                  | 0.7492 | 0.7752   | 0.7752      | 0.2911 | 0.2911 | 0.2584    | 0.3333 |
| CAM-DistilBERT | hatespeech | 64                  | 0.7781 | 0.7756   | 0.7736      | 0.2927 | 0.2908 | 0.5918    | 0.3341 |
| CAM-DistilBERT | hatespeech | 128                 | 0.8178 | 0.7764   | 0.7752      | 0.2974 | 0.2911 | 0.5255    | 0.3363 |
| CAM-DistilBERT | hatespeech | 256                 | 0.9173 | 0.7829   | 0.7752      | 0.3231 | 0.2911 | 0.5523    | 0.3492 |
| CAM-DistilBERT | hatespeech | 512                 | 1.1195 | 0.7986   | 0.7752      | 0.3789 | 0.2911 | 0.5652    | 0.3811 |
| CAM-BERT       | hatespeech | 4                   | 0.7745 | 0.7752   | 0.0533      | 0.2911 | 0.0346 | 0.2584    | 0.3333 |
| CAM-BERT       | hatespeech | 8                   | 0.7774 | 0.7752   | 0.0641      | 0.2911 | 0.0791 | 0.2584    | 0.3333 |
| CAM-BERT       | hatespeech | 16                  | 0.7256 | 0.7752   | 0.1795      | 0.2911 | 0.1041 | 0.2584    | 0.3333 |
| CAM-BERT       | hatespeech | 32                  | 0.7286 | 0.7752   | 0.7752      | 0.2911 | 0.2911 | 0.2584    | 0.3333 |
| CAM-BERT       | hatespeech | 64                  | 0.7478 | 0.7808   | 0.7752      | 0.3144 | 0.2911 | 0.5725    | 0.3448 |
| CAM-BERT       | hatespeech | 128                 | 0.7722 | 0.7982   | 0.7752      | 0.3739 | 0.2911 | 0.5922    | 0.3780 |
| CAM-BERT       | hatespeech | 256                 | 0.8349 | 0.8123   | 0.7752      | 0.4190 | 0.2911 | 0.5772    | 0.4081 |
| CAM-BERT       | hatespeech | 512                 | 0.9610 | 0.8212   | 0.7752      | 0.4492 | 0.2911 | 0.5587    | 0.4323 |
| CAM-RoBERTa    | hatespeech | 4                   | 0.7429 | 0.6545   | 0.5203      | 0.6112 | 0.3422 | 0.7334    | 0.6420 |
| CAM-RoBERTa    | hatespeech | 8                   | 0.7259 | 0.6807   | 0.4569      | 0.6523 | 0.4513 | 0.7365    | 0.6702 |
| CAM-RoBERTa    | hatespeech | 16                  | 0.7189 | 0.6832   | 0.7406      | 0.6555 | 0.7311 | 0.7384    | 0.6728 |
| CAM-RoBERTa    | hatespeech | 32                  | 0.6765 | 0.7272   | 0.6525      | 0.7108 | 0.5993 | 0.7695    | 0.7187 |
| CAM-RoBERTa    | hatespeech | 64                  | 0.6565 | 0.7787   | 0.4005      | 0.7705 | 0.3205 | 0.8066    | 0.7723 |
| CAM-RoBERTa    | hatespeech | 128                 | 0.6470 | 0.8262   | 0.6321      | 0.8219 | 0.5893 | 0.8466    | 0.8212 |
| CAM-RoBERTa    | hatespeech | 256                 | 0.6627 | 0.8752   | 0.5203      | 0.8732 | 0.3422 | 0.8882    | 0.8716 |
| CAM-RoBERTa    | hatespeech | 512                 | 0.7510 | 0.9025   | 0.5144      | 0.9013 | 0.4980 | 0.9119    | 0.8995 |
| CAM-DistilBERT | fake       | 4                   | 0.5797 | 0.7743   | 0.4619      | 0.7681 | 0.3484 | 0.7921    | 0.7920 |
| CAM-DistilBERT | fake       | 8                   | 0.5289 | 0.7960   | 0.4510      | 0.7920 | 0.4510 | 0.8088    | 0.7917 |
| CAM-DistilBERT | fake       | 16                  | 0.4695 | 0.8153   | 0.2307      | 0.8129 | 0.2140 | 0.8232    | 0.8120 |
| CAM-DistilBERT | fake       | 32                  | 0.4264 | 0.8277   | 0.2173      | 0.8258 | 0.2028 | 0.8339    | 0.8248 |
| CAM-DistilBERT | fake       | 64                  | 0.3885 | 0.8376   | 0.7371      | 0.8360 | 0.7364 | 0.8428    | 0.8350 |
| CAM-DistilBERT | fake       | 128                 | 0.3544 | 0.8515   | 0.4777      | 0.8503 | 0.3266 | 0.8551    | 0.8493 |
| CAM-DistilBERT | fake       | 256                 | 0.3286 | 0.8623   | 0.3084      | 0.8615 | 0.2493 | 0.8652    | 0.8605 |
| CAM-DistilBERT | fake       | 512                 | 0.3084 | 0.8748   | 0.6252      | 0.8740 | 0.6246 | 0.8774    | 0.8730 |
| CAM-BERT       | fake       | 4                   | 0.5689 | 0.7886   | 0.5045      | 0.7823 | 0.3579 | 0.8103    | 0.7830 |
| CAM-BERT       | fake       | 8                   | 0.5121 | 0.8173   | 0.4732      | 0.8133 | 0.4732 | 0.8333    | 0.8127 |
| CAM-BERT       | fake       | 16                  | 0.4402 | 0.8470   | 0.1881      | 0.8451 | 0.1744 | 0.8551    | 0.8439 |
| CAM-BERT       | fake       | 32                  | 0.3874 | 0.8603   | 0.2059      | 0.8588 | 0.1887 | 0.8674    | 0.8575 |
| CAM-BERT       | fake       | 64                  | 0.3422 | 0.8663   | 0.7881      | 0.8651 | 0.7875 | 0.8714    | 0.8639 |
| CAM-BERT       | fake       | 128                 | 0.3005 | 0.8817   | 0.3738      | 0.8810 | 0.2746 | 0.8844    | 0.8800 |
| CAM-BERT       | fake       | 256                 | 0.2700 | 0.8980   | 0.2703      | 0.8976 | 0.2206 | 0.8996    | 0.8967 |
| CAM-BERT       | fake       | 512                 | 0.2458 | 0.9094   | 0.5366      | 0.9091 | 0.5304 | 0.9104    | 0.9085 |
| CAM-RoBERTa    | fake       | 4                   | 0.5842 | 0.7916   | 0.2886      | 0.7836 | 0.2249 | 0.8225    | 0.7851 |
| CAM-RoBERTa    | fake       | 8                   | 0.5282 | 0.8505   | 0.1040      | 0.8477 | 0.1016 | 0.8649    | 0.8464 |
| CAM-RoBERTa    | fake       | 16                  | 0.4676 | 0.8817   | 0.8371      | 0.8801 | 0.8332 | 0.8923    | 0.8784 |
| CAM-RoBERTa    | fake       | 32                  | 0.3653 | 0.9144   | 0.8782      | 0.9136 | 0.8753 | 0.9202    | 0.9121 |
| CAM-RoBERTa    | fake       | 64                  | 0.2966 | 0.9297   | 0.3604      | 0.9293 | 0.3176 | 0.9330    | 0.9281 |
| CAM-RoBERTa    | fake       | 128                 | 0.2481 | 0.9431   | 0.6198      | 0.9428 | 0.5719 | 0.9448    | 0.9420 |
| CAM-RoBERTa    | fake       | 256                 | 0.2178 | 0.9564   | 0.5921      | 0.9563 | 0.4894 | 0.9573    | 0.9558 |
| CAM-RoBERTa    | fake       | 512                 | 0.2062 | 0.9653   | 0.6099      | 0.9652 | 0.5756 | 0.9656    | 0.9650 |
| CAM-DistilBERT | imdb       | 4                   | 0.5814 | 0.7939   | 0.4573      | 0.7936 | 0.3768 | 0.7953    | 0.7939 |
| CAM-DistilBERT | imdb       | 8                   | 0.5245 | 0.8058   | 0.5146      | 0.8057 | 0.4960 | 0.8066    | 0.8058 |
| CAM-DistilBERT | imdb       | 16                  | 0.5342 | 0.8242   | 0.8173      | 0.8242 | 0.8170 | 0.8243    | 0.8242 |
| CAM-DistilBERT | imdb       | 32                  | 0.4653 | 0.8414   | 0.8214      | 0.8414 | 0.8199 | 0.8415    | 0.8414 |
| CAM-DistilBERT | imdb       | 64                  | 0.4043 | 0.8599   | 0.3857      | 0.8599 | 0.2964 | 0.8601    | 0.8599 |
| CAM-DistilBERT | imdb       | 128                 | 0.3797 | 0.8368   | 0.4794      | 0.8368 | 0.3275 | 0.8368    | 0.8368 |
| CAM-DistilBERT | imdb       | 256                 | 0.3693 | 0.8421   | 0.2783      | 0.8421 | 0.2356 | 0.8421    | 0.8421 |
| CAM-DistilBERT | imdb       | 512                 | 0.3642 | 0.8458   | 0.6577      | 0.8458 | 0.6548 | 0.8458    | 0.8458 |
| CAM-BERT       | imdb       | 4                   | 0.5855 | 0.7926   | 0.4833      | 0.7925 | 0.3661 | 0.7932    | 0.7926 |
| CAM-BERT       | imdb       | 8                   | 0.5293 | 0.8051   | 0.5032      | 0.8051 | 0.4935 | 0.8054    | 0.8051 |
| CAM-BERT       | imdb       | 16                  | 0.4618 | 0.8220   | 0.2321      | 0.8221 | 0.2258 | 0.8221    | 0.8221 |
| CAM-BERT       | imdb       | 32                  | 0.4179 | 0.8326   | 0.2055      | 0.8326 | 0.1985 | 0.8326    | 0.8326 |
| CAM-BERT       | imdb       | 64                  | 0.3854 | 0.8409   | 0.7466      | 0.8409 | 0.7420 | 0.8409    | 0.8409 |
| CAM-BERT       | imdb       | 128                 | 0.3628 | 0.8478   | 0.4878      | 0.8478 | 0.3307 | 0.8478    | 0.8478 |
| CAM-BERT       | imdb       | 256                 | 0.3509 | 0.8522   | 0.2762      | 0.8522 | 0.2358 | 0.8522    | 0.8522 |
| CAM-BERT       | imdb       | 512                 | 0.3445 | 0.8566   | 0.6059      | 0.8566 | 0.6052 | 0.8566    | 0.8566 |
| CAM-RoBERTa    | imdb       | 4                   | 0.6162 | 0.8036   | 0.3504      | 0.8033 | 0.2822 | 0.8057    | 0.8036 |
| CAM-RoBERTa    | imdb       | 8                   | 0.5845 | 0.8091   | 0.3113      | 0.8091 | 0.2691 | 0.8092    | 0.8091 |
| CAM-RoBERTa    | imdb       | 16                  | 0.5342 | 0.8242   | 0.8173      | 0.8242 | 0.8170 | 0.8243    | 0.8242 |
| CAM-RoBERTa    | imdb       | 32                  | 0.4653 | 0.8414   | 0.8214      | 0.8414 | 0.8199 | 0.8415    | 0.8414 |
| CAM-RoBERTa    | imdb       | 64                  | 0.4043 | 0.8599   | 0.3858      | 0.8599 | 0.2964 | 0.8601    | 0.8599 |
| CAM-RoBERTa    | imdb       | 128                 | 0.3546 | 0.8738   | 0.4446      | 0.8737 | 0.3536 | 0.8740    | 0.8738 |
| CAM-RoBERTa    | imdb       | 256                 | 0.3177 | 0.8848   | 0.5143      | 0.8848 | 0.3649 | 0.8850    | 0.8848 |
| CAM-RoBERTa    | imdb       | 512                 | 0.2958 | 0.8932   | 0.4322      | 0.8932 | 0.3257 | 0.8934    | 0.8932 |

# Dense

| Model          | Dataset    | # Latent Dimensions | # Train Params | loss   | accuracy | cs accuracy | f1     | cs f1  | precision | recall |
|----------------|------------|---------------------|----------------|--------|----------|-------------|--------|--------|-----------|--------|
| DistilBERT     | hatespeech | N/A                 | 592899         | 0.5415 | 0.7966   | N/A         | 0.3718 | N/A    | 0.5759    | 0.3767 |
| CAM-DistilBERT | hatespeech | 250                 | 578253         | 0.5241 | 0.8152   | 0.1679      | 0.4300 | 0.0966 | 0.5661    | 0.4165 |
| BERT*          | hatespeech | N/A                 | 592899         | 0.4998 | 0.8220   | N/A         | 0.4496 | N/A    | 0.5659    | 0.4320 |
| CAM-BERT*      | hatespeech | 250                 | 578253         | 0.5166 | 0.8212   | 0.2502      | 0.4468 | 0.1596 | 0.5648    | 0.4299 |
| CAM-BERT*      | hatespeech | 300                 | 462002         | 0.5039 | 0.8241   | 0.1239      | 0.4543 | 0.1365 | 0.5678    | 0.4359 |
| RoBERTa        | hatespeech | N/A                 | 592899         | 0.5788 | 0.7849   | N/A         | 0.3292 | N/A    | 0.5815    | 0.3525 |
| CAM-RoBERTa    | hatespeech | 250                 | 578253         | 0.6843 | 0.7970   | 0.1691      | 0.3741 | 0.1309 | 0.5719    | 0.3781 |
| DistilBERT     | fake       | N/A                 | 592130         | 0.4916 | 0.7970   | N/A         | 0.7945 | N/A    | 0.8031    | 0.7939 |
| CAM-DistilBERT | fake       | 250                 | 385002         | 0.4501 | 0.8243   | 0.2530      | 0.8220 | 0.2525 | 0.8317    | 0.8210 |
| *BERT          | fake       | N/A                 | 592130         | 0.3750 | 0.8643   | N/A         | 0.8628 | N/A    | 0.8720    | 0.8614 |
| *CAM-BERT      | fake       | 250                 | 385002         | 0.4075 | 0.8569   | 0.6044      | 0.8550 | 0.5414 | 0.8663    | 0.8537 |
| *CAM-BERT      | fake       | 300                 | 462002         | 0.3851 | 0.8609   | 0.6163      | 0.8590 | 0.6058 | 0.8702    | 0.8577 |
| RoBERTa*       | fake       | N/A                 | 592899         | 0.3559 | 0.9089   | N/A         | 0.9079 | N/A    | 0.9170    | 0.9062 |
| CAM-RoBERTa*   | fake       | 250                 | 385002         | 0.3926 | 0.8975   | 0.6891      | 0.8963 | 0.6491 | 0.9063    | 0.8946 |
| DistilBERT     | imdb       | N/A                 | 592130         | 0.5127 | 0.7890   | N/A         | 0.7890 | N/A    | 0.7890    | 0.7890 |
| CAM-DistilBERT | imdb       | 10                  | 15402          | 0.4412 | 0.8211   | 0.3326      | 0.8211 | 0.2982 | 0.8212    | 0.8211 |
| BERT*          | imdb       | N/A                 | 592899         | 0.4165 | 0.8308   | N/A         | 0.8308 | N/A    | 0.8309    | 0.8308 |
| CAM-BERT*      | imdb       | 250                 | 385002         | 0.4373 | 0.8294   | 0.7499      | 0.8293 | 0.7444 | 0.8294    | 0.8294 |
| CAM-BERT*      | imdb       | 300                 | 462002         | 0.4148 | 0.8327   | 0.7190      | 0.8327 | 0.7187 | 0.8327    | 0.8327 |
| RoBERTa        | imdb       | N/A                 | 592899         | 0.4870 | 0.8367   | N/A         | 0.8367 | N/A    | 0.8368    | 0.8367 |
| CAM-RoBERTa    | imdb       | 250                 | 385002         | 0.4984 | 0.8324   | 0.2371      | 0.8324 | 0.2202 | 0.8325    | 0.8324 |
| CAM-RoBERTa    | imdb       | 300                 | 462002         | 0.4831 | 0.8382   | 0.5000      | 0.8382 | 0.3333 | 0.8381    | 0.8382 |
