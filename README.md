# AI System for Evidence-Grounded Defect Analysis in LPBF (Ti-6Al-4V)

## Overview
This project presents an early-stage prototype of an AI system for analysing defect formation in Laser Powder Bed Fusion (LPBF) of Ti-6Al-4V.

The system extracts, structures, and reasons over scientific literature to support engineering understanding of how process parameters influence defect formation.

---

## Problem
While large language models and search tools can retrieve and summarise scientific text, they do not explicitly capture relationships between process parameters, defect mechanisms, and material behaviour. They also do not connect evidence across multiple studies in a way that supports engineering reasoning.

As a result, engineers still rely heavily on manual interpretation to understand how processing conditions lead to specific defects.

---

## Solution
This project develops an evidence-grounded AI system that:

- Extracts entities and relationships from scientific literature  
- Builds a knowledge graph linking process parameters, mechanisms, and defects  
- Combines graph-based and semantic retrieval with large language models  
- Classifies scientific questions (e.g., explanation, comparison)  
- Generates structured, evidence-based answers grounded in retrieved literature  

---

## System Pipeline

1. Literature processing and text chunking  
2. Entity and relation extraction  
3. Knowledge graph construction  
4. Hybrid retrieval (graph + semantic search)  
5. Question classification and orchestration  
6. Evidence aggregation and answer generation  

---

## Example Query

Why does high laser power lead to keyhole porosity in LPBF?

---

## Example Output

**Question type:** Explanation  

**Matched graph nodes:**  
- laser power  
- keyhole  

**Top defect signals:**  
- porosity  
- keyhole  
- lack of fusion  

**Top process parameters:**  
- laser power  
- scan speed  

**Detected mechanism:**  
- keyhole instability  

**Graph path:**  
laser power → keyhole  

**Generated answer:**  
High laser power in LPBF can lead to keyhole porosity because it creates a deep melt pool and unstable vapour cavity dynamics. Collapse of the keyhole can trap gas and form pores. The retrieved evidence also indicates that porosity increases with increasing laser power and decreasing scan speed.

---

## Project Structure
