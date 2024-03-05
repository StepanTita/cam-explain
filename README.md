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

| Model            | # Latent Dims. | Metric | Top K | Jaccard | Mean Cosine | Min Cosine | Mean Euclid | Max Euclid | Accuracy | F1 Score | Precision | Recall |
|------------------|----------------|--------|-------|---------|-------------|------------|-------------|------------|----------|----------|-----------|--------|
| DistilBERT       | N/A            | Euclid | TOP 1 | 0.0817  | 0.9782      | 0.9782     | 3.1603      | 3.1602     | 0.0870   | 0.0457   | 0.1232    | 0.0799 |
|                  | N/A            |        | TOP 3 | 0.0752  | 0.9774      | 0.9766     | 3.2252      | 3.2796     | 0.1154   | 0.0872   | 0.1116    | 0.2054 |
|                  | N/A            |        | TOP 5 | 0.0705  | 0.9768      | 0.9758     | 3.2649      | 3.3371     | 0.1188   | 0.1009   | 0.0956    | 0.3069 |
| Space-DistilBERT | 3              |        | TOP 1 | 0.1519  | 0.8230      | 0.8230     | 3.7735      | 3.7735     | 0.1599   | 0.0967   | 0.1786    | 0.0977 |
|                  |                |        | TOP 3 | 0.1229  | 0.7967      | 0.7704     | 4.0580      | 4.2929     | 0.1871   | 0.1528   | 0.1393    | 0.2636 |
|                  |                |        | TOP 5 | 0.1098  | 0.7789      | 0.7394     | 4.2341      | 4.5597     | 0.1833   | 0.1616   | 0.1178    | 0.4037 |
| Space-DistilBERT | 64             |        | TOP 1 | 0.0416  | 0.7930      | 0.7930     | 9.9450      | 9.9451     | 0.0495   | 0.0276   | 0.0971    | 0.0466 |
|                  |                |        | TOP 3 | 0.0488  | 0.6889      | 0.6148     | 11.6900     | 12.8111    | 0.0782   | 0.0533   | 0.0915    | 0.1474 |
|                  |                |        | TOP 5 | 0.0500  | 0.6414      | 0.5481     | 12.4993     | 13.9659    | 0.0860   | 0.0630   | 0.0762    | 0.2388 |
| Space-DistilBERT | 128            |        | TOP 1 | 0.0374  | 0.8567      | 0.8567     | 11.7750     | 11.7750    | 0.0451   | 0.0256   | 0.0868    | 0.0446 |
|                  |                |        | TOP 3 | 0.0442  | 0.7363      | 0.6452     | 13.8267     | 15.3113    | 0.0711   | 0.0465   | 0.0767    | 0.1375 |
|                  |                |        | TOP 5 | 0.0433  | 0.6819      | 0.5718     | 14.7505     | 16.5501    | 0.0749   | 0.0495   | 0.0640    | 0.2151 |
| RoBERTa          | N/A            |        | TOP 1 | 0.1093  | 0.9957      | 0.9957     | 1.0776      | 1.0776     | 0.1078   | 0.0372   | 0.0534    | 0.0409 |
|                  | N/A            |        | TOP 3 | 0.0753  | 0.9954      | 0.9951     | 1.1125      | 1.1401     | 0.1139   | 0.0549   | 0.0443    | 0.1117 |
|                  | N/A            |        | TOP 5 | 0.0664  | 0.9952      | 0.9949     | 1.1324      | 1.1690     | 0.1115   | 0.0648   | 0.0445    | 0.1845 |
| Space-RoBERTa    | 3              |        | TOP 1 | 0.0561  | 0.9698      | 0.9698     | 1.5013      | 1.5013     | 0.0567   | 0.0282   | 0.0469    | 0.0347 |
|                  |                |        | TOP 3 | 0.0487  | 0.9657      | 0.9619     | 1.6005      | 1.6856     | 0.0746   | 0.0513   | 0.0450    | 0.1057 |
|                  |                |        | TOP 5 | 0.0471  | 0.9630      | 0.9579     | 1.6617      | 1.7725     | 0.0799   | 0.0601   | 0.0440    | 0.1780 |
| Space-RoBERTa    | 64             |        | TOP 1 | 0.1424  | 0.9932      | 0.9932     | 3.7781      | 3.7781     | 0.1361   | 0.0348   | 0.0489    | 0.0406 |
|                  |                |        | TOP 3 | 0.0749  | 0.9910      | 0.9889     | 4.2693      | 4.7223     | 0.1123   | 0.0516   | 0.0460    | 0.1081 |
|                  |                |        | TOP 5 | 0.0604  | 0.9896      | 0.9871     | 4.5777      | 5.1366     | 0.1014   | 0.0596   | 0.0447    | 0.1783 |
| Space-RoBERTa    | 128            |        | TOP 1 | 0.1454  | 0.9975      | 0.9975     | 3.6152      | 3.6153     | 0.1382   | 0.0356   | 0.0472    | 0.0412 |
|                  |                |        | TOP 3 | 0.0738  | 0.9967      | 0.9959     | 4.1352      | 4.6319     | 0.1106   | 0.0509   | 0.0435    | 0.1068 |
|                  |                |        | TOP 5 | 0.0603  | 0.9962      | 0.9953     | 4.4691      | 5.0729     | 0.1014   | 0.0604   | 0.0442    | 0.1800 |
| Model            | # Latent Dims. | Metric | Top K | Jaccard | Mean Cosine | Min Cosine | Mean Euclid | Max Euclid | Accuracy | F1 Score | Precision | Recall |
| DistilBERT       | N/A            | Cosine | TOP 1 | 0.      | 0.          | 0.         |             |            | 0.       | 0.       | 0.        | 0.     |
|                  | N/A            |        | TOP 3 | 0.      | 0.          | 0.         |             |            | 0.       | 0.       | 0.        | 0.     |
|                  | N/A            |        | TOP 5 | 0.      | 0.          | 0.         |             |            | 0.       | 0.       | 0.        | 0.     |
| Space-DistilBERT | 3              |        | TOP 1 | 0.      | 0.          | 0.         |             |            | 0.       | 0.       | 0.        | 0.     |
|                  |                |        | TOP 3 | 0.      | 0.          | 0.         |             |            | 0.       | 0.       | 0.        | 0.     |
|                  |                |        | TOP 5 | 0.      | 0.          | 0.         |             |            | 0.       | 0.       | 0.        | 0.     |
| Space-DistilBERT | 64             |        | TOP 1 | 0.0469  | 0.7973      | 0.7973     | 10.0647     | 10.0647    | 0.0557   | 0.0363   | 0.1352    | 0.0517 |
|                  |                |        | TOP 3 | 0.0562  | 0.6934      | 0.6242     | 11.8199     | 13.1033    | 0.0893   | 0.0699   | 0.1017    | 0.1593 |
|                  |                |        | TOP 5 | 0.0586  | 0.6475      | 0.5678     | 12.6531     | 14.4731    | 0.1000   | 0.0808   | 0.0823    | 0.2538 |
| Space-DistilBERT | 128            |        | TOP 1 | 0.0412  | 0.8594      | 0.8594     | 11.8854     | 11.8854    | 0.0494   | 0.0306   | 0.0934    | 0.0473 |
|                  |                |        | TOP 3 | 0.0477  | 0.7394      | 0.6517     | 13.9366     | 15.5507    | 0.0765   | 0.0557   | 0.0818    | 0.1421 |
|                  |                |        | TOP 5 | 0.0474  | 0.6851      | 0.5808     | 14.8602     | 16.9011    | 0.0817   | 0.0627   | 0.0721    | 0.2260 |
| RoBERTa          | N/A            |        | TOP 1 | 0.1185  | 0.9957      | 0.9957     | 1.0825      | 1.0825     | 0.1160   | 0.0362   | 0.0539    | 0.0404 |
|                  | N/A            |        | TOP 3 | 0.0776  | 0.9954      | 0.9952     | 1.1172      | 1.1522     | 0.1174   | 0.0553   | 0.0462    | 0.1143 |
|                  | N/A            |        | TOP 5 | 0.0665  | 0.9953      | 0.9950     | 1.1369      | 1.1863     | 0.1117   | 0.0627   | 0.0441    | 0.1835 |
| Space-RoBERTa    | 3              |        | TOP 1 | 0.0620  | 0.9699      | 0.9699     | 1.5032      | 1.5032     | 0.0621   | 0.0290   | 0.0462    | 0.0350 |
|                  |                |        | TOP 3 | 0.0510  | 0.9657      | 0.9621     | 1.6017      | 1.6889     | 0.0779   | 0.0516   | 0.0447    | 0.1067 |
|                  |                |        | TOP 5 | 0.0486  | 0.9630      | 0.9581     | 1.6628      | 1.7772     | 0.0822   | 0.0605   | 0.0439    | 0.1784 |
| Space-RoBERTa    | 64             |        | TOP 1 | 0.1430  | 0.9932      | 0.9932     | 3.7865      | 3.7865     | 0.1362   | 0.0341   | 0.0459    | 0.0397 |
|                  |                |        | TOP 3 | 0.0756  | 0.9911      | 0.9890     | 4.2769      | 4.7416     | 0.1133   | 0.0512   | 0.0462    | 0.1078 |
|                  |                |        | TOP 5 | 0.0610  | 0.9897      | 0.9871     | 4.5844      | 5.1620     | 0.1022   | 0.0595   | 0.0451    | 0.1785 |
| Space-RoBERTa    | 128            |        | TOP 1 | 0.1445  | 0.9976      | 0.9976     | 3.6521      | 3.6521     | 0.1377   | 0.0338   | 0.0451    | 0.0391 |
|                  |                |        | TOP 3 | 0.0747  | 0.9967      | 0.9960     | 4.1702      | 4.7152     | 0.1120   | 0.0523   | 0.0458    | 0.1092 |
|                  |                |        | TOP 5 | 0.0605  | 0.9963      | 0.9954     | 4.5010      | 5.1834     | 0.1016   | 0.0599   | 0.0443    | 0.1789 |