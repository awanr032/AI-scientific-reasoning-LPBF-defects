# AI System for Defect Analysis in LPBF (Ti-6Al-4V)

## Overview
This repository contains an early-stage prototype of an AI system for analysing defect formation in Laser Powder Bed Fusion (LPBF), with a focus on Ti-6Al-4V.

The goal of this work is to move beyond simple retrieval and build a system that can extract, organise, and reason over scientific knowledge to support understanding of how process parameters influence defect formation.

---

## Motivation
Understanding defect formation in additive manufacturing typically requires manually reviewing and comparing multiple research papers. While existing tools can retrieve and summarise text, they do not explicitly capture relationships between process parameters, mechanisms, and defects, nor do they connect evidence across studies in a structured way.

This project is an attempt to bridge that gap by combining knowledge extraction, graph-based retrieval, and structured reasoning.

---

## What the system does
At a high level, the system:

- Processes scientific literature related to LPBF  
- Extracts entities such as process parameters, defects, and mechanisms  
- Builds a knowledge graph linking these concepts  
- Retrieves relevant evidence using both graph structure and semantic similarity  
- Uses a language model to generate answers grounded in retrieved evidence  

In addition, an agent-based pipeline is used to:
- classify the type of question  
- aggregate evidence across chunks  
- identify dominant defects, parameters, and mechanisms  
- discover graph paths between relevant concepts  
- generate a structured report rather than only free-text output  

---

## System pipeline
The core pipeline consists of:

1. Data loading and preprocessing (not included in this repo)  
2. Semantic indexing of text chunks  
3. Query normalization  
4. Graph-based retrieval  
5. Evidence aggregation  
6. Question classification  
7. Graph path discovery  
8. Answer generation using a language model  
9. Structured report generation  

---

## Example query
Why does high laser power lead to keyhole porosity in LPBF?

---

## Example output

Question type: Explanation  

Top defects:
- porosity  
- keyhole  

Top parameters:
- laser power  
- scan speed  

Mechanism:
- keyhole instability  

Graph path:
laser power → keyhole  

Answer:
High laser power can create a deeper melt pool and unstable vapour cavity. When the keyhole becomes unstable or collapses, gas can get trapped, forming pores. The retrieved evidence also suggests that porosity tends to increase with higher laser power and lower scan speed.

---

## Evaluation
A small benchmark of domain-specific questions is used to evaluate how well the system captures relationships between process parameters and defects.

The evaluation includes:
- retrieval accuracy  
- defect label precision and recall  
- parameter identification accuracy  
- response latency  

The results show that the system can reliably identify dominant defect signals and relevant parameters across different types of questions.

---
## How to run this file
- The data has been processed as json chunks. Upload the file"Revised_rag_chunks(1).json" file on google colab or jupyter notebook.

## Repository Structure

- AI_Defect_Reasoning_LPBF.ipynb  
  Main notebook implementing the Graph RAG pipeline for defect reasoning.

- sample_data/  
  Contains small example data for demonstration:
  - sample_raw_data.pdf → Example research paper input  
  - sample_cleaned_data.docx → Preprocessed/cleaned text  
  - sample_chunks.json → Extracted entities and relationships in JSON format  


