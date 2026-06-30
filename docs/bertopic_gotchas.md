# BERTopic Gotchas

## **Overview table**

| Area | Gotcha (short) |
| ----- | ----- |
| Embeddings | Not precomputing; wrong doc length |
| Dimensionality reduction | UMAP randomness; wrong metric |
| Clustering / topics | Misusing `nr_topics`; HDBSCAN prediction quirks |
| Representations / labeling | Stopword removal timing; overtrusting default words |
| Serialization / inference | `.transform` behavior after `safetensors` |
| Scaling / performance | Huge c-TF‑IDF matrix; bad `min_df`/`ngram_range` |
| RAPIDS specific | Install order; normalization; `.transform` limitations |

---

## **General BERTopic gotchas**

1. **Not precomputing embeddings before tuning**  
   * Many users call `BERTopic()` and let it compute embeddings every time they tweak parameters, which explodes run time and hides what’s actually changing.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
   * Best practice is to compute embeddings once (e.g. using sentence‑transformers) and pass `embeddings` to `.fit_transform` so you can iterate on UMAP/HDBSCAN/vectorizer/representation settings independently.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
2. **Feeding full long documents instead of sentences/paragraphs**  
   * Default sentence‑transformers work best at sentence/short paragraph scale; long documents are truncated internally and you silently lose most of the text.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * A common fix is splitting large documents into sentences/paragraphs before embedding, or using `.approximate_distribution` to get document‑level topic mixtures after training on shorter segments.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
3. **Assuming UMAP is deterministic without `random_state`**  
   * UMAP is stochastic; if you don’t set `random_state`, the reduced embeddings and clusters shift subtly across runs.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
   * You should always pass a UMAP instance with a fixed `random_state` into BERTopic when you care about reproducibility or stable topic IDs.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
4. **Misusing `nr_topics` instead of HDBSCAN parameters**  
   * `nr_topics` merges topics *after* clustering; it does not directly control cluster formation and can produce odd merges or a confusing topic structure.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
   * Better practice is to control number of topics via the cluster model (default HDBSCAN) using `min_cluster_size` and related parameters, then optionally reduce topics later.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
5. **Ignoring HDBSCAN outliers and their impact**  
   * HDBSCAN assigns some documents to topic `-1` (outliers); beginners are surprised to see many `-1` labels and try to force everything into a topic.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
   * If you later use `.reduce_outliers`, you must be careful when also doing topic reduction/merging, because remapping `-1` to several topics can make it unclear how those documents should be handled during merges.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
6. **Stopword removal at the wrong stage**  
   * Removing stopwords before embedding can hurt transformer embeddings because they depend on full context.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * You should remove or down‑weight stopwords after clustering, using `CountVectorizer(stop_words="english")`, `ClassTfidfTransformer(reduce_frequent_words=True)`, or representation models like KeyBERTInspired/MMR.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
7. **Over‑aggressive vocabulary (`min_df`, `ngram_range`) on large data**  
   * Setting `min_df` too low or `ngram_range` too wide (e.g. `(1, 3)`) on hundreds of thousands of documents can lead to an enormous c‑TF‑IDF matrix and memory issues.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
   * On large datasets, it’s advised to set `min_df` to at least around 10 and keep `ngram_range` to `(1, 1)` or `(1, 2)` to control vocabulary size.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
8. **Misunderstanding topic representations vs clustering**  
   * Changing `vectorizer_model`, stopword settings, or representation models does not alter clusters; it only changes how topics are described.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * People sometimes expect representation tweaks to “fix bad topics,” but they actually need to adjust embeddings, UMAP, or clustering for structural changes.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
9. **Relying solely on the default c‑TF‑IDF words**  
   * The default top‑n words can be redundant or overly dominated by frequent terms, making topics look worse than they actually are.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * Using multi‑aspect representations (KeyBERTInspired, POS, MMR, OpenAI labels) often gives much more interpretable labels and richer understanding.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
10. **Serialization gotchas (`pickle` vs `safetensors`)**  
    * When you save with `safetensors`, the dimensionality reduction and clustering models are *not* stored; `.transform` then uses cosine similarity against topic embeddings instead of UMAP+HDBSCAN.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
    * This makes inference much faster but can change topic assignments for new documents compared to a fully serialized model (e.g. `pickle`), which can surprise users expecting identical behavior.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
11. **Transforming with different embedding models than training**  
    * If you load a safetensors model and plug in a different embedding model, topic embeddings and document embeddings live in different spaces and assignments become unreliable.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
    * You must ensure that the same embedding model (or at least identical output space) is used when training and when doing inference.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
12. **Visualization overinterpretation (2D UMAP plots)**  
    * People over‑interpret 2D document plots as “the truth”; but you’re compressing hundreds of dimensions into 2, which is only an approximation.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
    * Use these plots to get intuition, not to make fine‑grained judgments about topic boundaries or cluster separations.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
13. **Scaling to very large datasets without incremental strategy**  
    * Running BERTopic on millions of documents with default settings can hit memory and performance limits, especially in UMAP, HDBSCAN, and c‑TF‑IDF.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
    * The author suggests starting with subsets, raising `min_df`, limiting `ngram_range`, precomputing/saving embeddings, and ensuring up‑to‑date HDBSCAN for \>500k records.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]

---

## **RAPIDS (cuML) \+ BERTopic gotchas**

1. **Install order and environment compatibility**  
   * Installing BERTopic and cuML in one `pip` command often fails due to CUDA dependency resolution; you should install cuML first, then BERTopic.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * You need the cuML build matching your CUDA version (e.g. `cuml-cu12`, `cuml-cu13`), otherwise GPU acceleration will not work or will crash at runtime.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
2. **Forgetting to normalize embeddings before GPU UMAP/HDBSCAN**  
   * cuML UMAP defaults to Euclidean distance; if embeddings are unnormalized, distances reflect magnitude as well as semantic direction, which can distort clusters.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * Normalizing embeddings (e.g. with `cuml.preprocessing.normalize`) makes Euclidean distance behave like cosine similarity over unit vectors, which is typically what you want for semantic topics.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
3. **Assuming GPU UMAP is deterministic without `random_state`**  
   * Just like CPU UMAP, cuML UMAP is stochastic unless you fix `random_state`; failing to do so leads to run‑to‑run variation in layout and clusters.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * You should pass a cuML UMAP instance with explicit `random_state` into BERTopic if reproducibility matters.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
4. **Topic‑document probabilities for `.transform` not supported (older cuML/HDBSCAN)**  
   * At least as of BERTopic v0.13, topic‑document probability matrices for *unseen* data via `.transform` are not available when using cuML’s HDBSCAN; only training data probabilities (`.fit`, `.fit_transform`) are fully supported.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * Users expecting `.transform` probabilities on new documents with GPU HDBSCAN may need CPU HDBSCAN, safetensors‑based cosine inference, or approximate distributions instead.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
5. **Expecting strict determinism from cuML HDBSCAN `transform`**  
   * HDBSCAN’s prediction step uses approximate algorithms and may not match training cluster assignments exactly, especially for points near boundaries; this holds for CPU and GPU variants.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
   * Even with fixed seeds, you should expect small differences in topic assignments for some edge‑case documents when using `.transform` on GPU.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
6. **Mixing CPU and GPU components incorrectly**  
   * Passing a CPU UMAP with a cuML HDBSCAN (or vice versa) can negate performance gains or introduce data transfer overheads and subtle compatibility issues.\[[linkedin](https://www.linkedin.com/posts/nicholasebecker_faster-topic-modeling-with-bertopic-and-rapids-activity-7022243478067671040-ixHA)\]  
   * For best performance, keep both dimensionality reduction and clustering consistently on GPU (cuML UMAP \+ cuML HDBSCAN) or consistently on CPU.\[[linkedin](https://www.linkedin.com/posts/nicholasebecker_faster-topic-modeling-with-bertopic-and-rapids-activity-7022243478067671040-ixHA)\]  
7. **Overlooking GPU memory limits with huge embedding sets**  
   * Large corpora with high‑dimensional embeddings can quickly exhaust GPU memory in cuML UMAP/HDBSCAN.\[[linkedin](https://www.linkedin.com/posts/nicholasebecker_faster-topic-modeling-with-bertopic-and-rapids-activity-7022243478067671040-ixHA)\]  
   * Practical mitigations include using smaller embedding models, batching, subsampling for initial experiments, and careful tuning of `n_neighbors`, `n_components`, and `min_cluster_size`.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
8. **Not leveraging precomputed embeddings with RAPIDS**  
   * Some users move to RAPIDS expecting “automatic” speedups but still recompute embeddings each run, leaving the main bottleneck untouched.\[[linkedin](https://www.linkedin.com/posts/nicholasebecker_faster-topic-modeling-with-bertopic-and-rapids-activity-7022243478067671040-ixHA)\]  
   * Precomputing embeddings on CPU or GPU and then using cuML just for UMAP/HDBSCAN usually yields the biggest speed gains.\[[linkedin](https://www.linkedin.com/posts/nicholasebecker_faster-topic-modeling-with-bertopic-and-rapids-activity-7022243478067671040-ixHA)\]  
9. **Assuming safetensors inference is GPU‑accelerated clustering**  
   * When you load a safetensors model, the clustering and UMAP are removed; `.transform` relies on cosine similarity between document and topic embeddings, not cuML HDBSCAN.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
   * People sometimes think they are still “using RAPIDS for inference,” but the speedup mainly comes from bypassing clustering entirely.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
10. **Ignoring cuML version differences and feature gaps**  
    * cuML’s UMAP/HDBSCAN APIs and capabilities evolve; some examples in older docs (e.g. probability support) may not behave identically in your installed version.\[[linkedin](https://www.linkedin.com/posts/nicholasebecker_faster-topic-modeling-with-bertopic-and-rapids-activity-7022243478067671040-ixHA)\]  
    * You should check RAPIDS/cuML release notes and verify that required features (e.g. specific parameters, prediction options) exist for your version.

---

## **BERTopic modeling / workflow gotchas (independent of RAPIDS)**

These are more about how you *use* BERTopic.

1. **Not tuning cluster granularity (`min_cluster_size`)**  
   * Using default HDBSCAN settings can produce many micro‑topics or a tiny number of broad topics that are hard to interpret.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
   * Adjusting `min_cluster_size` upward reduces micro‑clusters and yields broader topics; lowering it yields more fine‑grained topics.\[[maartengr.github](https://maartengr.github.io/BERTopic/faq.html)\]  
2. **Skipping iterative evaluation on small subsets**  
   * Running full‑scale models first and only then inspecting topic quality wastes time and makes it harder to diagnose issues.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
   * Best practice is to develop/tune on subsets (e.g. 50k–100k documents), then scale up once hyperparameters and representations behave well.\[[github](https://github.com/MaartenGr/BERTopic/issues/491)\]  
3. **Assuming model is “done” after `.fit_transform`**  
   * Many powerful adjustments happen *after* training: `.update_topics`, custom labels, multi‑aspect representations, outlier reduction, and approximate distributions.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * Treat BERTopic more like a pipeline you refine than a one‑shot black box.  
4. **Comparing models with different embedding spaces**  
   * Comparing topic embeddings across BERTopic models trained with different embedding models leads to meaningless similarity scores.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * For cross‑model topic similarity, you must use the *same* embedding model across all BERTopic instances.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
5. **Overloading the model with multimodal or metadata embeddings without clear design**  
   * BERTopic lets you cluster on arbitrary embeddings (e.g. image features, metadata), but if you mix modalities naively, topics become uninterpretable.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]  
   * When doing multimodal or metadata‑driven topics, design carefully which embeddings represent what, and keep text for c‑TF‑IDF representations aligned with the clustered features.\[[github](https://github.com/MaartenGr/BERTopic/discussions/2124)\]

