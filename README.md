# singlecell_gpt

Pretraining single-cell LLMs.


# Survey and/or References


***Review***

- Chen Liang, 2023, Github: https://github.com/dzyim/single-cell-models
- Best practices for single-cell analysis across modalities, [Web]( https://www.nature.com/articles/s41576-023-00586-w )
- Zhang, S., Fan, R., Liu, Y., Chen, S., Liu, Q., & Zeng, W. (2023). Applications of transformer-based language models in bioinformatics: a survey. Bioinformatics Advances, 3(1), vbad001. [Paper](https://academic.oup.com/bioinformaticsadvances/article/3/1/vbad001/6984737 )
- Ji, Y., Lotfollahi, M., Wolf, F. A., & Theis, F. J. (2021). Machine learning for perturbational single-cell omics. Cell Systems, 12(6), 522-537.



***Foundation models***

- **scEval**: Liu, T., Li, K., Wang, Y., Li, H., & Zhao, H. (2023). Evaluating the Utilities of Large Language Models in Single-cell Data Analysis. bioRxiv, 2023-09.
- Chen, J., Xu, H., Tao, W., Chen, Z., Zhao, Y., & Han, J. D. J. (2023). Transformer for one stop interpretable cell type annotation. Nature Communications, 14(1), 223. [Web](https://www.nature.com/articles/s41467-023-35923-4 )
- **CellPLM**: Wen, H., Tang, W., Dai, X., Ding, J., Jin, W., Xie, Y., & Tang, J. (2023). CellPLM: Pre-training of Cell Language Model Beyond Single Cells. bioRxiv, 2023-10. [Web](https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1.abstract ) ; [GitHub](https://github.com/OmicsML/CellPLM) ; [Supplementary](https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1.supplementary-material ) 
- **scGPT** (Cited by 25): Cui, H., Wang, C., Maan, H., Pang, K., Luo, F., & Wang, B. (2023). scgpt: Towards building a foundation model for single-cell multi-omics using generative ai. bioRxiv, 2023-04. [GitHub]( https://github.com/bowang-lab/scGPT ); [PDF](https://www.biorxiv.org/content/10.1101/2023.04.30.538439v2.full.pdf ); 
- **xTrimoGene**: Gong, J., Hao, M., Cheng, X., Zeng, X., Liu, C., Ma, J., ... & Song, L. (2023). xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data. arXiv preprint arXiv:2311.15156. [API]( https://api.biomap.com/xTrimoGene/apply )
- **scFoundation**: Hao, M., Gong, J., Zeng, X., Liu, C., Guo, Y., Cheng, X., ... & Zhang, X. (2023). Large Scale Foundation Model on Single-cell Transcriptomics. bioRxiv, 2023-05. [Web]( https://www.biorxiv.org/content/10.1101/2023.05.29.542705v4.abstract ); [GitHub]( https://github.com/biomap-research/scFoundation )
- **GeneFormer**: Theodoris, C. V., Xiao, L., Chopra, A., Chaffin, M. D., Al Sayed, Z. R., Hill, M. C., ... & Ellinor, P. T. (2023). Transfer learning enables predictions in network biology. Nature, 1-9. [Web]( https://www.nature.com/articles/s41586-023-06139-9 ); [Code]( https://huggingface.co/ctheodoris/Geneformer ); 
- **tGPT**: Shen, H., Liu, J., Hu, J., Shen, X., Zhang, C., Wu, D., ... & Li, X. (2023). Generative pretraining from large-scale transcriptomes for single-cell deciphering. Iscience, 26(5). [PDF]( https://www.cell.com/iscience/pdf/S2589-0042(23)00613-2.pdf )
- **scBERT** (Nature Machine Intelligence; 2018): Yang, F., Wang, W., Wang, F., Fang, Y., Tang, D., Huang, J., ... & Yao, J. (2022). scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data. Nature Machine Intelligence, 4(10), 852-866. [Paper]: https://www.nature.com/articles/s42256-022-00534-z



***Perturbation Prediction***

- **CellOT** (**NatureMethods**): Bunne, C., Stark, S. G., Gut, G., Del Castillo, J. S., Levesque, M., Lehmann, K. V., ... & Rätsch, G. (2023). Learning single-cell perturbation responses using neural optimal transport. Nature methods, 1-10.
- **GERAS** (**NatureBiotechnology**; **JureLeskovec**): Roohani, Y., Huang, K., & Leskovec, J. (2023). Predicting transcriptional outcomes of novel multigene perturbations with gears. Nature Biotechnology, 1-9. [PDF]( https://www.nature.com/articles/s41587-023-01905-6 ); [GitHub] (https://github.com/snap-stanford/GEARS )

(Drug Perturbation Prediction)

- Dong, M., Wang, B., Wei, J., de O. Fonseca, A. H., Perry, C. J., Frey, A., ... & van Dijk, D. (2023). Causal identification of single-cell experimental perturbation effects with CINEMA-OT. Nature Methods, 1-11. [Web](https://www.nature.com/articles/s41592-023-02040-5) , [GitHub](https://github.com/vandijklab/CINEMA-OT ), [Perturbation Analysis in the scverse ecosystem.](https://github.com/theislab/pertpy )

- **CPA** (**Cited by 28**; Facebook;) Lotfollahi, M., Klimovskaia Susmelj, A., De Donno, C., Hetzel, L., Ji, Y., Ibarra, I. L., ... & Theis, F. J. (2023). Predicting cellular responses to complex perturbations in high‐throughput screens. Molecular Systems Biology, e11517. [Web](https://www.embopress.org/doi/full/10.15252/msb.202211517 ) [GitHub](https://github.com/theislab/cpa ) [GitHub-reproducibility](https://github.com/theislab/cpa-reproducibility/tree/main/notebooks)

- - CPA (Cited by 28; Facebook;): Lotfollahi, M., Susmelj, A. K., De Donno, C., Ji, Y., Ibarra, I. L., Wolf, F. A., ... & Lopez-Paz, D. (2021). Learning interpretable cellular responses to complex perturbations in high-throughput screens. BioRxiv, 2021-04. [GitHub]( http://github.com/facebookresearch/CPA )

- **OntoVAE**: Doncevic, D., & Herrmann, C. (2023). Biologically informed variational autoencoders allow predictive modeling of genetic and drug-induced perturbations. Bioinformatics, 39(6), btad387. [Web](https://academic.oup.com/bioinformatics/advance-article-abstract/doi/10.1093/bioinformatics/btad387/7199588 ) [GitHub](https://github.com/hdsu-bioquant/onto-vae )

- **DREEP**: Pellecchia, S., Viscido, G., Franchini, M., & Gambardella, G. (2023). Predicting drug response from single-cell expression profiles of tumours. BMC medicine, 21(1), 476. [Web](https://link.springer.com/article/10.1186/s12916-023-03182-1 ), [GitHub]( https://github.com/gambalab/DREEP )

- **chemCPA**  (**Neurips**): Hetzel, L., Böhm, S., Kilbertus, N., Günnemann, S., Lotfollahi, M., & Theis, F. (2022). Predicting single-cell perturbation responses for unseen drugs. arXiv preprint arXiv:2204.13545.，[Web](https://arxiv.org/abs/2204.13545 ) [Supplementary](https://proceedings.neurips.cc/paper_files/paper/2022/file/aa933b5abc1be30baece1d230ec575a7-Supplemental-Conference.pdf ); [GitHub](https://github.com/theislab/chemCPA ) 

- - **chemCPA**: Hetzel, L., Boehm, S., Kilbertus, N., Günnemann, S., & Theis, F. (2022). Predicting cellular responses to novel drug perturbations at a single-cell resolution. Advances in Neural Information Processing Systems, 35, 26711-26722. [PDF](https://openreview.net/pdf?id=vRrFVHxFiXJ ) [GitHub](https://github.com/theislab/chemCPA )

- **scGEN** (**NatureMethods**; **Cited by 260**): Lotfollahi, M., Wolf, F. A., & Theis, F. J. (2019). scGen predicts single-cell perturbation responses. Nature methods, 16(8), 715-721. [Web]( https://www.nature.com/articles/s41592-019-0494-8 ); [Code]( https://github.com/theislab/scgen ) & [ Code of paper ]( https://github.com/theislab/scgen-reproducibility )



***RLHF***

- ChatGPT: https://openai.com/blog/chatgpt
- InstructGPT: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35, 27730-27744. [Web]( https://arxiv.org/abs/2203.02155 )
- **trl**: Train transformer language models with reinforcement learning. [GitHub](https://github.com/huggingface/trl ), [Docs](https://huggingface.co/docs/trl/index )
- trlX: trlX: A scalable framework for RLHF; [GitHub](https://github.com/CarperAI/trlx )
- RL4LMs: Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization. [Web](https://arxiv.org/abs/2210.01241 );  [GitHub](https://github.com/allenai/RL4LMs）
- **Transformers**: State-of-the-Art Natural Language Processing, [GitHub](https://github.com/huggingface/transformers ); [Web]([huggingface.co/transformers](https://huggingface.co/transformers) )



***Molecule Generation***

- **GexMolGen** (Gene Expression-based Molecule Generator): Cross-modal Generation of Hit-like Molecules via Foundation Model Encoding of Gene Expression Signatures, [Web](https://www.biorxiv.org/content/10.1101/2023.11.11.566725v2 )  



# Code

- CPA: [combosciplex-tutorial]( https://cpa-tools.readthedocs.io/en/latest/tutorials/combosciplex.html )

- Pertpy: Perturbation Analysis in the scverse ecosystem. https://pertpy.readthedocs.io/en/latest/

- - scGen - Perturbation response prediction,[scgen_perturbation_prediction](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/scgen_perturbation_prediction.html) , [Colab](https://colab.research.google.com/github/theislab/scgen/blob/master/docs/tutorials/scgen_perturbation_prediction.ipynb)

- scGEN, Single cell perturbation prediction, https://github.com/theislab/scgen/ , https://scgen.readthedocs.io/en/stable/tutorials/scgen_perturbation_prediction.html, 

- Transformers: Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. [huggingface.co/transformers](https://huggingface.co/transformers)

- TRL: TRL is a full stack library where we provide a set of tools to train transformer language models with Reinforcement Learning, from the Supervised Fine-tuning step (SFT), Reward Modeling step (RM) to the Proximal Policy Optimization (PPO) step. https://huggingface.co/docs/trl/index

- scVI Tools  (NatureMethods; Cited by 1215): Lopez, R., Regier, J., Cole, M. B., Jordan, M. I., & Yosef, N. (2018). Deep generative modeling for single-cell transcriptomics. Nature methods, 15(12), 1053-1058. [GitHub](https://github.com/scverse/scvi-tools)