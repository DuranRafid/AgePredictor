# Predicting Chronological Age from Single Cell RNA-seq of Whole Lung Dissociates

This work was done as a CPCB rotation project with Prof. Ziv Bar-Joseph.  

Aging is a vital aspect of human life. Understanding the mechanism of aging on cellular and genomic level is very important to diagnosis and treatment of many diseases, but far from achieved. One very recent study shows that Reversal of biological clock restores vision in old mice [1]. By understanding proper genetic mechanisms of aging in humans, we may lead to similar breakthroughs for human treatments too. 

### Related Works 

Previous works of predicting human age mainly used DNA methylation data [2,3], some also used blood cytology data [4]. Peters et. al. [5] used both transcriptomic and methylation data of human peripheral blood to find the markers of aging. The most recent work predicting age only from human gene expression data used an ensemble of LDA classifiers on transcriptome of human dermal fibroblast [6]. They applied their methods on  dermal bulk rna seq data of 133 people aged between 1 to 94 years. Their results show that their proposed method outperforms all the previous methods in predicting age from transcriptome data. To the best of our knowledge no work has been done yet to predict age from single cell transcriptomic data. 

## Our Approach

### Dataset

The original dataset available in [GEO GSE136831](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE136831) contains scRNA-seq data of 312928 cells across 45,947 genes from the whole lung disassociates of 78 subjects (28 control, 32 IPF, and 18 COPD). Further information regarding the dataset can be found at the paper [7].  In this work of predicting age, we only selected the cells from control subjects assuming it is more probable that the biological age of their lungs matches with their chronological ages. Selecting only the control subjects and removing the cells identified as multiplets by the authors resulted in 95303 cells. The aging information along with sex is also obtained from GEO GSE136831 and available in the dataset folder of the repository. The 28 control subjects are aged between 20 to 80 years old.

### Quality Control

The quality control steps (removing cells with too few genes, high mitochondrial reads, low mapped reads, etc.) was performed by the authors of the dataset [7].

### Preprocessing

After that, we used `scanpy` to filter out cells with less than 200 gene UMIs and genes expressed in less than 3 cells. All the 95303 cells retained after this filtering process, but for genes, 38645 remained and others were discarded. This led to a dataset of 38645 genes x 95303 cells, which further went through a scaled normalization of 10000 UMIs per cell. The dataset matrix was further natural log-transformed with a pseudo count of 1, which is available in the dataset folder along with metadata of the genes and the cells.  

From these single-cell rna seq data matrix, pseudo bulk rna seq was generated by averaging the amount of UMIs across the genes for each subject and was further normalized by 1 million UMIs per subject to make the numbers comparable to TPM (Transcripts per Million). Further cell identity-wise pseudo bulk rna seq was generated following similar processes. Cell identity information was collected from the control_df available in GEO GSE136831. These identities were determined by the authors using Louvain Cluster analysis in the Seurat R package. They determined a total of 37 cell identities comprising 5 major cell categories: 'Myeloid', 'Lymphoid', 'Endothelial', 'Stromal', 'Epithelial'.

 
### Feature Selection 

Selecting the appropriate features or genes was a very hard problem in the context of this work. Here we adopted one of the two following approaches: 
Selected top N highly variable genes using method `seurat` from `scanpy`, where N $\in$ {10, 50, 100, 1000}
Selected those genes having expression level greater than T1 for at least one subject and having a fold change of at least T2 between any two samples, where T1 and T2 are two threshold hyperparameters (This was used in the work [6]) 
For some of the methods, we further reduced dimension using PCA and used the top 5 or 10 PCA components as features. 

### Methods/ Experiments

We applied several simple regressors on the selected features including two Bayesian models, e.g., Gaussian Process Regression, Automatic Relevance Detection Regression (a form of Bayesian Ridge Regression), Uni/Bi/Tri Linear Layer Elastic Net regression implemented using PyTorch, Support Vector Regressions, Random Forest Regression, etc. We also applied the ‘Ensemble of LDA classifier’ method proposed by [6] on our pseudo-bulk rna seq data, but the performance was very poor (25.8 MED:30.0 R2:-2.03). Increasing the number of linear layers to more than two in Elastic Net drops the performance, so we did not try adding more than three layers. Most of methods were implemented with default parameter setting available in package `sklearn`.

We apply these methods on both our pseudo-bulk rna seq data and cell-type based rna seq data. Method trained on a cell type-specific rna seq data is viewed as an ‘Expert’ of that type. We also experimented with ensemble regressors of the cell type ‘Experts’. There were no subjects with all the cell types present in it, so outputs from the ‘Experts’ of those cell types are set as zero. 

### Results

Some cell type experts perform better than the overall bulk-rna seq, but none of them perform very satisfactorily. Due to the very small number of samples and very high dimensional data, most of the methods did not fit well. 

<img style="float: left" src="/best_results/fig_plot_predictions_VE_Capillary_ARandomForestRegressor.png" title="Best Predictor" height="300" width="300"> 

The best-fitted model is RandomForestRegressor on VE Capillary A Cell Expert having goodness of fit (measured in R squared) of 0.64 and MAE and MDE of 8.6 and 6.4 respectively. From a goodness of fit perspective, the second-best is a Gaussian Process Regressor with a linear combination of DotProduct and White Kernel on VE Capillary B Cell Expert ( R squared: 0.44, MAE: 11.4, MDE: 11.7). The following best performer is ARDRegressors on VE Peribronchial (R squared: 0.33, MAE: 12.3, MDE: 10.1). However, none of them can predict age for all individuals as the cell types are not present in everyone and none of them are good enough, a clue can be found that VE (Venus Endothelial) cells may have something to do with aging. This is a very interesting direction that needs further investigation. Also, whereas using all cell experts in ensembling does not accrue well. good ensemble methods may be designed using a subset of cell experts. Bayesian methods also seem promising as they can also give some uncertainty related to their prediction, which is very much relevant in predicting age. But as the number of samples are too low in our case, the predictions of bayesian models (mean posterior distribution) depend largely on the prior, choosing proper priors for model parameters in this case is also a direction worth exploring. All of these are left as further works.

## References 

[1] H. Ledford, “Reversal of biological clock restores vision in old mice,” Nature News, 02-Dec-2020. [Online]. Available: https://www.nature.com/articles/d41586-020-03403-0. [Accessed: 07-Dec-2020] 

[2] Horvath, Steve. "DNA methylation age of human tissues and cell types." Genome biology 14.10 (2013): 3156.

[3] Hannum, Gregory, et al. "Genome-wide methylation profiles reveal quantitative views of human aging rates." Molecular cell 49.2 (2013): 359-367.

[4] Putin, Evgeny, et al. "Deep biomarkers of human aging: application of deep neural networks to biomarker development." Aging (Albany NY) 8.5 (2016): 1021.

[5] Peters, Marjolein J., et al. "The transcriptional landscape of age in human peripheral blood." Nature communications 6.1 (2015): 1-14.

[6] Fleischer, Jason G., et al. "Predicting age from the transcriptome of human dermal fibroblasts." Genome biology 19.1 (2018): 221.

[7] Adams, Taylor S., et al. "Single-cell RNA-seq reveals ectopic and aberrant lung-resident cell populations in idiopathic pulmonary fibrosis." Science advances 6.28 (2020): eaba1983.



