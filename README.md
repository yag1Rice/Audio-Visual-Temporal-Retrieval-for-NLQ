[Audio-Visual Temporal Retrieval for NLQ.md](https://github.com/user-attachments/files/29442332/Audio-Visual.Temporal.Retrieval.for.NLQ.md)
# Audio-Visual Temporal Retrieval for Natural Language Queries

## Overview

This project investigates multimodal retrieval for natural language queries (NLQ) involving human interactions in egocentric video. The system combines audio and visual retrieval pipelines to identify relevant temporal segments within long-form videos and evaluates whether late-fusion approaches can improve retrieval performance.

The primary focus is on queries containing conversational interaction of the following forms:

* "Who did I talk to in location X?"
* "When did I talk to a person with role X?"

The retrieval framework leverages separate embedding spaces for audio and visual modalities and explores multiple fusion strategies to combine their predictions.


## Methodology

### 1. Multi-Scale Temporal Pyramid Construction

For each video, a temporal pyramid is constructed using overlapping segments of varying durations:

* Segment lengths: 1–10 seconds
* Overlap: 75%
* Stored representation:

  ```
  (video_id, start_time, end_time)
  ```

Only timestamps are stored to maintain scalability and avoid pre-extracting clips.

---

### 2. Feature Extraction

#### Audio Pipeline

Audio corresponding to each temporal segment is extracted and encoded using CLAP.

* Feature Extractor: CLAP
* Similarity Metric: Cosine Similarity
* Indexing: FAISS

The resulting embeddings are stored in a dedicated audio retrieval index.

#### Video Pipeline

Visual content from each temporal segment is encoded using:

* Baseline: VideoPrism
* Extension: Qwen3-VL-Embedding

Embeddings are stored in a separate FAISS index for visual retrieval.

* Similarity Metric: Cosine Similarity

---

### 3. Query Encoding

For each natural language query:

#### Audio Query Encoder

The query is embedded using the CLAP text encoder.

#### Video Query Encoder

The query is embedded using:

* VideoPrism text encoder (baseline)
* Qwen3-VL-Embedding text encoder (extension)

These retrieval pipelines operate independently within their respective embedding spaces.

The queries are reformulated using Gemini to transform questions into declarative captions before encoding.

---

### 4. Independent Retrieval

For each query:

1. Retrieve top-K segments from the audio FAISS index.
2. Retrieve top-K segments from the video FAISS index.

This produces two ranked retrieval lists.

---

### 5. Late Fusion

Two fusion strategies are investigated.

#### Reciprocal Rank Fusion (RRF)

Audio and video rankings are combined using weighted RRF:

```
RRF = α × Audio_RRF + (1 − α) × Video_RRF
```

#### Min-Max Score Normalization

Similarity scores from each modality are normalized and combined through weighted averaging:

```
Normalized_Score = α × Normalized_Audio_Score + (1 − α) × Normalized_Video_Score
```

---

### 6. Hyperparameter Search

Fusion weights are evaluated using grid search:

```
α ∈ {0.05, 0.10, 0.15, ..., 0.95}
```

The goal is to determine the optimal contribution of audio and visual modalities.

---

### 7. Temporal Non-Maximum Suppression

To reduce redundant overlapping predictions, both standard Temporal NMS and Soft-NMS are evaluated.

#### Overlap Metrics

Two overlap metrics are explored:

##### Intersection over Union (IoU)

```
IoU = Intersection / Union
```

##### Intersection over Minimum (IoM)

```
IoM = Intersection / min(segment_length_1, segment_length_2)
```


#### Standard NMS

Segments with overlap above a threshold are removed, retaining only the highest-scoring prediction.

#### Soft-NMS

Instead of suppressing overlapping segments, Soft-NMS progressively decays their confidence scores, allowing highly-overlapping yet potentially relevant predictions to remain in the ranked list.

The following Soft-NMS variants are implemented and evaluated:

- **Linear Soft-NMS (IoU)**
- **Gaussian Soft-NMS (IoU)**
- **Linear Soft-NMS (IoM)**
- **Gaussian Soft-NMS (IoM)**

This allows comparison between different overlap definitions (IoU vs. IoM) and score decay strategies (linear vs. Gaussian) for temporal retrieval.


## Evaluation

Three retrieval systems are evaluated independently:

1. Audio-only retrieval
2. Video-only retrieval
3. Audio-visual fusion retrieval

### Metrics

* Recall@3
* Recall@5
* IoU Threshold = 0.3
* IoU Threshold = 0.5
* Mean IoU (mIoU)

## Results

### VideoPrism

#### Best Single-Modality Results

| Metric                                           |      Score |
| ------------------------------------------------ | ---------: |
| Audio Recall@5 (NMS, IoM, IoU=0.3)               | **0.1273** |
| Audio Recall@5 (Gaussian Soft-NMS, IoM, IoU=0.3) | **0.1273** |
| Audio Recall@5 (Gaussian Soft-NMS, IoU, IoU=0.3) | **0.1273** |

#### Best RRF Fusion Results

| Metric                                         |      Score |
| ---------------------------------------------- | ---------: |
| Recall@5 (RRF, α=0.35, Gaussian Soft-NMS, IoM) | **0.1455** |
| Recall@5 (RRF, α=0.40, NMS, IoM)               | **0.1455** |
| Recall@5 (RRF, α=0.40, Gaussian Soft-NMS, IoM) | **0.1455** |
| Recall@5 (RRF, α=0.45, NMS, IoU)               | **0.1455** |
| Recall@5 (RRF, α=0.45, NMS, IoM)               | **0.1455** |

#### Best Min-Max Fusion Results

| Metric                                             |      Score |
| -------------------------------------------------- | ---------: |
| Recall@5 (Min-Max, α=0.15, NMS, IoM)               | **0.1273** |
| Recall@5 (Min-Max, α=0.15, Gaussian Soft-NMS, IoM) | **0.1273** |
| Recall@5 (Min-Max, α=0.15, Gaussian Soft-NMS, IoU) | **0.1273** |
| Recall@5 (Min-Max, α=0.20, NMS, IoM)               | **0.1273** |

---

### Qwen3-VL Embedding Extension

#### Best Single-Modality Results

| Metric                                           |      Score |
| ------------------------------------------------ | ---------: |
| Audio Recall@5 (NMS, IoM, IoU=0.3)               | **0.1273** |
| Audio Recall@5 (Gaussian Soft-NMS, IoM, IoU=0.3) | **0.1273** |
| Audio Recall@5 (Gaussian Soft-NMS, IoU, IoU=0.3) | **0.1273** |

#### Best RRF Fusion Results

| Metric                                         |      Score |
| ---------------------------------------------- | ---------: |
| Recall@5 (RRF, α=0.55, NMS, IoU)               | **0.1273** |
| Recall@5 (RRF, α=0.60, Linear Soft-NMS, IoU)   | **0.1273** |
| Recall@5 (RRF, α=0.60, Gaussian Soft-NMS, IoU) | **0.1273** |
| Recall@5 (RRF, α=0.95, Gaussian Soft-NMS, IoU) | **0.1273** |
| Recall@5 (RRF, α=0.95, Gaussian Soft-NMS, IoM) | **0.1273** |

#### Best Min-Max Fusion Results

| Metric                                             |      Score |
| -------------------------------------------------- | ---------: |
| Recall@5 (Min-Max, α=0.40, NMS, IoU)               | **0.2000** |
| Recall@5 (Min-Max, α=0.40, Linear Soft-NMS, IoU)   | **0.2000** |
| Recall@5 (Min-Max, α=0.45, Gaussian Soft-NMS, IoM) | **0.1818** |
| Recall@5 (Min-Max, α=0.50, Gaussian Soft-NMS, IoM) | **0.1818** |
| Recall@5 (Min-Max, α=0.55, Gaussian Soft-NMS, IoU) | **0.1818** |


## Limitations

While the proposed retrieval framework demonstrates the potential of multimodal fusion for temporal natural language retrieval, several limitations remain:

* **Limited Dataset Size.** Experiments were conducted on only **55 natural language queries (NLQs)** from the EGO4D validation set. This relatively small evaluation subset limits the statistical significance of the reported results and may not fully represent the diversity of the benchmark.

* **No Separate Validation Set for Hyperparameter Tuning.** Fusion weights (α) were selected using the same evaluation set on which performance was reported. A dedicated train/validation split for hyperparameter selection would reduce the risk of overfitting and provide a more reliable estimate of generalization.

* **Fixed Temporal Segmentation.** Candidate segments are generated using a fixed temporal pyramid with predefined window lengths (1–10 seconds). This may fail to accurately capture events that are substantially shorter, longer, or span multiple temporal windows.

* **Computational Cost of Embedding Extraction.** Feature extraction is computationally expensive, particularly for **Qwen3-VL-2B**, which significantly increases preprocessing time. Additionally, visual embeddings are generated from only **16 sampled frames** per segment, potentially missing important temporal information.

* **Query Reformulation Dependency.** Natural language queries are reformulated into declarative captions using Gemini before retrieval. The overall retrieval performance therefore depends on the quality and consistency of these reformulated queries, introducing an additional source of variability.

