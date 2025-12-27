# PRD.md - Product Requirements Document
# Iris Species Classification: SVM vs QP-Solver Approach

**Version:** 2.0
**Date:** December 2025
**Project:** L23 - Machine Learning Classification Comparison
**Course:** AI Developer Expert - Optimization Methods

---

## Project Overview

- **Project Name:** Iris Species Classification - sklearn SVM vs Custom QP-Solver Implementation
- **One-Line Description:** Educational comparison of scikit-learn's optimized SVM against a hand-built Quadratic Programming solver, demonstrating the mathematical equivalence of both approaches on the classic Iris dataset.
- **Problem Statement:** How do we find the optimal separating hyperplane between classes, and how does a production-grade SVM library compare to solving the underlying QP problem directly?
- **Why This Matters:** Understanding the mathematical foundations (inner products, hyperplanes, margin maximization, Lagrange multipliers) transforms SVM from a "black box" into a transparent, interpretable algorithm.

---

## Educational Objectives

### What Students Will Learn

1. **Inner Product as Decision Rule**
   - The dot product `<x, w>` tells us which side of a hyperplane a point lies on
   - Positive = one class, Negative = other class, Zero = on the boundary

2. **The Three Mathematical Tricks** (from class L23)
   - **Trick 1:** Linearizing constraints by multiplying by label `y_i`
   - **Trick 2:** Canonical hyperplane (force margin = 1 for closest points)
   - **Trick 3:** Convert max(1/||w||) to min(||w||²) for QP formulation

3. **Support Vectors**
   - Only the points ON the margin boundary matter
   - The hyperplane is fully determined by these "support vectors"

4. **Primal vs Dual Formulation**
   - Primal: Optimize w and b directly
   - Dual: Optimize Lagrange multipliers α (enables kernel trick)

---

## Target Users & Applications

### Primary Users
- ML students learning SVM mathematical foundations
- Developers wanting to understand what sklearn does internally
- Researchers needing interpretable classification

### Use Cases

1. **Educational Demonstration**
   - Scenario: Student wants to understand how SVM really works
   - Benefit: See the math translate directly to code and results

2. **Algorithm Validation**
   - Scenario: Verify that custom QP solver produces same results as sklearn
   - Benefit: Confidence in mathematical understanding

3. **Botanical Classification**
   - Scenario: Classify iris species from petal/sepal measurements
   - Benefit: Classic ML benchmark with known expected results

---

## Functional Requirements

### Method 1: scikit-learn SVM (sklearn.svm.SVC)

| Feature | Description |
|---------|-------------|
| Kernel Options | Linear (for comparison) and RBF (for best accuracy) |
| Multi-class Strategy | One-vs-One (default in sklearn) |
| Output | Accuracy, confusion matrix, support vectors, decision boundaries |
| Parameters | C (regularization), kernel type, gamma |

### Method 2: Custom QP-Solver SVM (cvxopt)

| Feature | Description |
|---------|-------------|
| Formulation | Primal QP: min ½||w||² s.t. y_i(w·x_i + b) ≥ 1 |
| Solver | cvxopt.solvers.qp (interior point method) |
| Multi-class | One-vs-Rest (OvR) with 3 binary classifiers |
| Output | Weight vectors w, bias b, identified support vectors |

### Hierarchical Classification Option
Following the YarinPerez approach:
- **Level 1:** Separate Setosa (easy) from Versicolor+Virginica
- **Level 2:** Separate Versicolor from Virginica (harder, overlapping)
- This demonstrates varying classification difficulty

### Comparison Features

| Comparison Metric | Description |
|-------------------|-------------|
| Weight Vector Similarity | Cosine similarity between sklearn and QP weights |
| Support Vector Match | Do both methods identify the same critical points? |
| Decision Boundary | Visual overlay of both methods' boundaries |
| Training Time | sklearn (LIBSVM) vs cvxopt performance |
| Accuracy | Both should achieve >95% on linear-separable pairs |

---

## Technical Requirements

### Environment
- **Python Version:** 3.10+
- **Virtual Environment:** UV (MANDATORY per guidelines)
- **Operating Systems:** Windows (WSL), Linux, macOS

### Core Dependencies

```
numpy>=1.24.0          # Array operations (NO basic Python loops!)
pandas>=2.0.0          # Data loading
scikit-learn>=1.3.0    # Reference SVM implementation
cvxopt>=1.3.0          # QP solver for custom SVM
matplotlib>=3.7.0      # Visualization
seaborn>=0.12.0        # Enhanced plots
```

### Code Quality Requirements (per PROJECT_GUIDELINES.md)

| Requirement | Value |
|-------------|-------|
| Max lines per file | **150** |
| Type hints | **Mandatory** on all functions |
| Docstrings | **Mandatory** with WHY explanations |
| Comments | Explain to 15-year-old level |
| Relative paths | **Mandatory** (use pathlib) |
| Ring buffer logging | **Mandatory** |

### Performance Requirements
- Complete pipeline: < 30 seconds
- Memory usage: < 500 MB
- Reproducible results (fixed random_state=42)

---

## Mathematical Background

### The SVM Optimization Problem

**Naive Formulation (Horrifyingly Nonlinear):**
```
max   min  |<w, x_i> + b|
w,b    i      ||w||

subject to: sign(<w, x_i> + b) = sign(y_i)
```

**After Three Tricks → Clean QP:**
```
min   ½||w||²
w,b

subject to: y_i(<w, x_i> + b) ≥ 1  for all i
```

### Trick 1: Linearize Constraints
- Problem: `sign()` function is nonlinear
- Solution: Multiply both sides by y_i ∈ {-1, +1}
- Result: `y_i · (<w, x_i> + b) ≥ 0` (always positive when correct)

### Trick 2: Canonical Hyperplane
- Problem: Infinite equivalent hyperplanes (w, 2w, 100w all same boundary)
- Solution: Scale w so closest point has functional margin = 1
- Result: `min y_i(<w, x_i> + b) = 1` eliminates the inner min

### Trick 3: Convert to Minimization
- Problem: max(1/||w||) is awkward
- Solution: Equivalent to min(||w||), which equals min(½||w||²)
- Result: Quadratic objective → standard QP form

### QP Standard Form (for cvxopt)
```
min   ½ xᵀHx + fᵀx
 x

subject to: Ax ≤ b
```

**Mapping SVM to QP:**
| SVM | QP Parameter |
|-----|--------------|
| (w, b) | x (decision variables) |
| ½\|\|w\|\|² | ½xᵀHx where H = I (identity) |
| None | f = 0 (no linear term) |
| y_i(w·x_i + b) ≥ 1 | -y_i(w·x_i + b) ≤ -1 → Ax ≤ b |

### Support Vectors
Points where `y_i(<w, x_i> + b) = 1` exactly (on the margin boundary).
These are the ONLY points that determine the optimal hyperplane.

---

## Dataset Specification

### Iris Dataset (Data/Iris-Species/Iris.csv)

| Property | Value |
|----------|-------|
| Samples | 150 (50 per class) |
| Features | 4: SepalLength, SepalWidth, PetalLength, PetalWidth |
| Classes | 3: Iris-setosa, Iris-versicolor, Iris-virginica |
| Separability | Setosa linearly separable; Versicolor/Virginica overlap |

### Class Distribution Analysis
- **Setosa:** Clearly separable (small petals)
- **Versicolor:** Overlaps with Virginica
- **Virginica:** Overlaps with Versicolor

This makes it perfect for demonstrating:
- Easy binary classification (Setosa vs others)
- Harder binary classification (Versicolor vs Virginica)
- Multi-class strategies (OvR, OvO)

---

## Success Criteria

### Functional Success
- [ ] sklearn SVM achieves >95% accuracy
- [ ] Custom QP solver achieves >90% accuracy
- [ ] Weight vectors have cosine similarity >0.95 (for linear kernel)
- [ ] Support vectors identified correctly
- [ ] All visualizations generated

### Code Quality Success
- [ ] All Python files ≤ 150 lines
- [ ] Every function has type hints and docstring
- [ ] Ring buffer logging implemented
- [ ] All paths relative (pathlib)
- [ ] Passes fresh venv test

### Documentation Success
- [ ] README explains math to 15-year-old level
- [ ] 3 complete test runs documented with visuals
- [ ] Code files table with line counts
- [ ] Virtual environment setup instructions

---

## Visualization Requirements

### Required Graphs (minimum 5)

1. **Decision Boundary Comparison** (2D PCA projection)
   - sklearn boundary vs QP solver boundary
   - Support vectors highlighted

2. **Confusion Matrices** (side-by-side)
   - sklearn results
   - QP solver results

3. **Support Vector Visualization**
   - Which points are support vectors?
   - Compare between methods

4. **Feature Importance** (weight vector analysis)
   - Bar chart of |w| components
   - Which features matter most?

5. **Accuracy Comparison Bar Chart**
   - Per-class and overall accuracy
   - sklearn vs QP solver

6. **Training Time Comparison**
   - Bar chart: sklearn vs cvxopt

---

## Constraints & Assumptions

### Constraints
- Must use provided Iris.csv dataset
- Max 150 lines per Python file
- UV virtual environment required
- All paths must be relative

### Assumptions
- Linear kernel sufficient for comparison (RBF optional)
- Data is already clean (no missing values)
- Binary classification forms basis for multi-class

---

## References

### Course Materials
- L23 Slides: "SVM Primal Problem and QP as Optimization Problem"
- L23 Book: "Binary Classification using Support Vector Machine"

### External References
- Andrew Ng's CS229 SVM Notes
- cvxopt documentation: https://cvxopt.org/
- sklearn SVC: https://scikit-learn.org/stable/modules/svm.html

### Reference Implementations (for learning, not copying)
- github.com/YarinPerez/SVM (hierarchical approach)
- github.com/alienspirit7/L23_HomeWork (optimization comparison)
- github.com/Moshe710B/AI_DEV_Course_L23_QP-sklearn_SVM (unit testing)

---

## Deliverables Summary

| Deliverable | Location |
|-------------|----------|
| Main entry point | main.py |
| sklearn SVM module | src/sklearn_svm.py |
| QP Solver SVM module | src/qp_svm.py |
| Multi-class wrapper | src/multiclass.py |
| Data loader | src/data_loader.py |
| Visualizations | src/visualizer.py |
| Metrics | src/metrics.py |
| Path utilities | src/utils/paths.py |
| Ring buffer logger | src/utils/logger.py |
| Results | results/examples/run_{1,2,3}/ |
| Graphs | results/graphs/ |

---

**Document Status:** Ready for Approval
**Next Step:** Review tasks.json, then await approval before Phase 2 implementation
