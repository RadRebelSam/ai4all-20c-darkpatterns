# Dark Pattern Recognition

This project uses machine learning to detect deceptive dark-pattern language in e-commerce website text. It compares text classification approaches and documents the ethical tradeoffs of using AI to flag potentially manipulative design.

**GitHub Page:** https://radrebelsam.github.io/ai4all-20c-darkpatterns/

## Research Question

Can machine learning models detect dark-pattern language in e-commerce website text, and which categories of dark patterns are hardest to classify?

## Project Evolution

The project began with a broad survey of multiple dark-pattern datasets from Kaggle. Early attempts to combine several category-labeled datasets ran into inconsistent label schemas and imbalanced class distributions that inflated accuracy scores without producing a reliable classifier. After comparing label quality and coverage across four candidate datasets, we converged on a single balanced binary dataset of 2,356 examples (1,178 dark pattern, 1,178 not) as the primary training source and used secondary category-labeled datasets only for exploratory analysis.

The modeling pipeline went through two main iterations. The first used a simple bag-of-words (CountVectorizer) approach and showed that even basic text features could achieve over 85% accuracy. The second switched to TF-IDF with unigram and bigram features, which improved both precision and F1 score by giving more weight to distinctive terms rather than raw counts. Five classifiers were compared on an 80/20 stratified split. Logistic Regression emerged as the best performer at 93.9% accuracy and 0.936 F1, offering a good balance of accuracy, speed, and interpretability through its learned feature coefficients.

## Design Decisions and Reasoning

This section documents the main project decisions and why we made them.

| Decision | What we did | Reasoning |
| --- | --- | --- |
| Topic selection | Focused on dark-pattern recognition instead of AI job replacement. | Dark-pattern datasets were more directly usable for text classification and allowed a concrete demo with website text. |
| Dataset strategy | Used the Krish Uppal balanced dataset as the primary baseline; used Devitachi and Akash Nath as expanded/secondary sources. | The primary dataset has a clean 50/50 binary label split. The other datasets add source coverage, but they have different schemas, overlap, and class imbalance, so they are better for robustness checks than the default app model. |
| Common schema | Normalized different datasets into shared fields such as `text`, `label`, `category`, and `source`. | Each dataset has different columns. A common schema lets the same training and evaluation code work across sources. |
| Train/test split | Used an 80/20 random stratified split with `random_state=42`. | Random splitting avoids row-order bias. Stratification keeps the dark-pattern/not-dark-pattern ratio similar in train and test sets, making the evaluation fairer. |
| Feature method | Used `TF-IDF` with unigrams and bigrams. | TF-IDF is not the classifier itself; it turns text into numeric features. Bigrams help capture phrases such as `limited time` and `only left`. |
| Model comparison | Tested Logistic Regression, Linear SVM, Naive Bayes, Decision Tree, and Random Forest. | Comparing several models gives a stronger baseline than using only one model. Logistic Regression had the best primary F1 score and gives confidence scores for the app/scanner. |
| Demo approach | Built both a Streamlit app and a Playwright webpage scanner. | Streamlit is easy to explain for manual examples. Playwright makes the demo feel closer to real webpage scanning by collecting visible page text automatically. |
| Confidence threshold | The Playwright scanner shows dark-pattern results above a default 65% confidence threshold. | This hides weaker predictions during the demo and keeps attention on higher-confidence examples. Confidence is model certainty, not a guarantee that the prediction is correct. |
| Discount false-positive filter | Suppressed plain price/discount snippets unless they include pressure language. | Real sale pages often include harmless text like `Previous price: $99.00 47% off`. The filter reduces obvious false positives while still allowing pressure phrases like `Hurry, 47% off ends soon`. |
| Product-title false-positive filter | Suppressed catalog-style product titles/spec snippets unless they include pressure language. | Live tests showed false positives on product titles with model numbers, storage sizes, camera specs, or labels like `US Stock`. These are usually catalog metadata, not manipulative language. |
| Sentiment/context decision | Documented sentiment and UI context as future work instead of adding them now. | Sentiment can help with shame/fear/guilt language, but dark patterns depend heavily on button roles, popup placement, timing, and user flow. Adding this now would require new labeled data and a more complex pipeline. |
| Production limitation | Documented OCR, screenshot, color/layout, DOM/CSS, and UX-flow detection as future production layers. | Even with Playwright, this project is text-first. A real production detector would need additional evidence for image text, visual hierarchy, hidden links, checkout friction, cancellation difficulty, and other non-text patterns. |

## Algorithm: TF-IDF + Logistic Regression

The pipeline has two stages:

1. **TF-IDF Vectorizer** — converts each text snippet into a numeric vector. Terms that appear frequently in dark-pattern examples but rarely in neutral text receive high weights. Unigrams and bigrams are both included (`ngram_range=(1,2)`), so phrases like "limited time" are treated as a single feature.

2. **Logistic Regression** — learns a weight for each TF-IDF feature. Positive weights push the prediction toward "dark pattern"; negative weights push toward "not dark pattern." The learned coefficients are directly interpretable as feature importance scores.

## Model Comparison

All five models were evaluated on the same 20% held-out test set (stratified split, `random_state=42`).

| Model | Accuracy | Precision | Recall | F1 | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| **Logistic Regression** | **0.939** | **0.973** | **0.903** | **0.936** | Best overall; interpretable coefficients |
| Linear SVM | 0.932 | 0.959 | 0.903 | 0.930 | Close second; no probability output |
| Naive Bayes | 0.928 | 0.928 | 0.928 | 0.928 | Fast; assumes feature independence |
| Random Forest | 0.928 | 0.986 | 0.869 | 0.923 | High precision but lower recall |
| Decision Tree | 0.917 | 0.980 | 0.852 | 0.912 | Prone to overfitting on text features |

**Why Logistic Regression?** It achieves the highest F1 score, produces calibrated probability estimates (enabling confidence scores in the demo app), and its coefficients can be inspected to verify that the model is learning semantically meaningful features rather than spurious correlations. Linear SVM scores nearly as well but does not natively output probabilities.

## Dataset

- **Primary (binary):** Krish Uppal balanced dataset — 2,356 examples, 50/50 split between dark-pattern and not-dark-pattern text
- **Categories in dark-pattern class:** Scarcity (418), Social Proof (312), Urgency (210), Misdirection (195), Obstruction (27), Sneaking (12), Forced Action (4)
- **Source:** Kaggle, English-language e-commerce websites

## Visualizations

Publication-quality figures are in [`reports/figures/`](reports/figures/) and embedded in the GitHub Page.

| Figure | Description |
| --- | --- |
| `model_comparison.png` | Grouped horizontal bar chart comparing all 5 models across 4 metrics |
| `confusion_matrix.png` | Heatmap for Logistic Regression on the 20% test set |
| `category_distribution.png` | Dark-pattern category counts showing class imbalance within categories |
| `top_features.png` | Top 15 TF-IDF coefficients showing which words drive dark-pattern predictions |

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models and save metrics:

```bash
python scripts/train_models.py
```

Train the expanded dataset experiment that also incorporates Devitachi and Akash Nath sources:

```bash
python scripts/train_models.py --dataset expanded
```

The expanded option writes `reports/model_metrics_expanded.csv` and `artifacts/best_binary_model_expanded.joblib`. The default app still uses the primary Logistic Regression model because it is balanced, easy to explain, and provides confidence scores.

Generate visualizations:

```bash
python scripts/generate_visualizations.py
```

Run tests:

```bash
python -m pytest
```

Run the Streamlit demo:

```bash
streamlit run app/streamlit_app.py
```

Scan a live webpage with Playwright:

```bash
python -m playwright install chromium
python scripts/scan_page.py https://example.com
```

The scanner opens the page, waits briefly for dynamic content or popups, extracts visible body text, splits it into short snippets, and runs the trained text model over each snippet. This is a demo-friendly webpage scanner, not full visual UI understanding.

By default, the Streamlit app and Playwright scanner also apply two lightweight demo safeguards:

- They suppress simple price/discount snippets such as `sale price $10.00` or `Previous price: $99.00 47% off` unless the snippet includes pressure language like `hurry`, `limited time`, `ends soon`, or `only 2 left`.
- They suppress catalog-style product title/spec snippets such as model numbers, storage sizes, camera/lens specs, and labels like `US Stock` unless the snippet includes pressure language.

These post-processing filters reduce obvious false positives in the live demo. A longer-term model improvement would be to add more normal sale/discount and product-catalog examples labeled as `Not Dark Pattern` and retrain.

Sentiment and context are important future considerations. Sentiment could help identify emotionally manipulative language such as guilt or fear, but it would not be enough by itself because dark patterns often depend on where the text appears, what button it is attached to, and what action the user is being pushed toward. For this version, we keep the model focused on text classification and treat richer UI context, screenshot features, DOM structure, and sentiment features as future work rather than adding extra training complexity now.

For a real production detector, Playwright text extraction would only be one layer. A stronger system could capture full-page screenshots and run OCR to recover text embedded in images, banners, or canvas elements. It could also use computer-vision features to analyze button color contrast, visual hierarchy, disabled-looking choices, hidden links, and misleading emphasis. To detect UX-flow dark patterns, the system would need to record multi-step interactions such as signup, checkout, cancellation, and unsubscribe flows, then compare how easy the preferred business action is versus the user-protective action.

To show the raw model behavior without this filter:

```bash
python scripts/scan_page.py https://www.ebay.com/deals --no-price-filter --no-product-filter
```

## Responsible AI

This model should be used as a review aid rather than a final judgment. Dark patterns can depend on full page layout, timing, checkout flow, and visual design; this version only analyzes text. A false positive from this model could expose a legitimate business to unwarranted scrutiny; a false negative could leave a consumer unprotected. Model predictions should be one input into a larger review process, not a standalone verdict.

Key limitations:
- Training data is English-language, Western-focused e-commerce; the model may not generalize to other languages, regions, or domains
- Some categories (Obstruction, Sneaking, Forced Action) have very few examples and are more likely to be misclassified
- Text-only features cannot capture visual manipulation, flow-based dark patterns, or context-dependent intent
- The Playwright scanner can often capture popup text when it appears in the DOM, but it may miss cross-origin iframes, image/canvas text, closed shadow DOM content, or websites that block automation
- The Streamlit/scanner price-discount filter is a rule-based demo safeguard, not a substitute for better labeled training data
- The product-title/spec filter is also a rule-based demo safeguard; it can reduce catalog false positives but should not be treated as proof that a page has no dark patterns
- Sentiment alone cannot determine whether a design is deceptive; the model would need surrounding UI context and human review for borderline cases
- Screenshot + OCR could recover image-based text, but OCR can misread stylized fonts, low-contrast text, animations, and responsive layouts
- Color and layout analysis would require computer-vision or DOM/CSS features, not just TF-IDF text features
- Production UX-flow detection would require scripted user journeys and event logs to measure friction, repeated interruptions, hidden fees, cancellation difficulty, and forced-choice paths

## Next Steps

- Add cross-lingual examples and evaluate generalization beyond English e-commerce
- Extend to multi-class classification to identify which *type* of dark pattern is present
- Integrate visual features (screenshot-based detection) to catch non-textual dark patterns
- Explore richer context features, such as nearby button text, popup/modal location, DOM structure, sentiment, and checkout-step information
- Add screenshot + OCR experiments for image-based text and visual banners
- Add flow-based evaluation for checkout, subscription, cancellation, and unsubscribe journeys
- Evaluate on real-world web-scraped data rather than curated datasets
- Partner with consumer advocacy groups to validate findings in deployment

## Citations

1. Mathur, A., Acar, G., Friedman, M. J., Lucherini, E., Mayer, J., Chetty, M., & Narayanan, A. (2019). Dark Patterns at Scale: Findings from a Crawl of 11K Shopping Websites. *Proceedings of the ACM on Human-Computer Interaction*, 3(CSCW), Article 81. https://doi.org/10.1145/3359183

2. Brignull, H. (2023). Deceptive Patterns. *deceptive.design*. https://www.deceptive.design/

3. Gray, C. M., Kou, Y., Battles, B., Hoggatt, J., & Toombs, A. L. (2018). The Dark (Patterns) Side of UX Design. *Proceedings of the 2018 CHI Conference on Human Factors in Computing Systems*, Paper 534. https://doi.org/10.1145/3173574.3174108

4. Federal Trade Commission. (2022). Bringing Dark Patterns to Light. *FTC Staff Report*. https://www.ftc.gov/reports/bringing-dark-patterns-light

5. European Parliament. (2022). Digital Services Act. *Regulation (EU) 2022/2065*. https://eur-lex.europa.eu/eli/reg/2022/2065/oj

6. Luguri, J. & Strahilevitz, L. J. (2021). Shining a Light on Dark Patterns. *Journal of Legal Analysis*, 13(1), 43–109. https://doi.org/10.1093/jla/laaa006

7. Narayanan, A., Mathur, A., Chetty, M., & Kshirsagar, M. (2020). Dark Patterns: Past, Present, and Future. *Queue*, 18(2), 67–92. https://doi.org/10.1145/3400899.3400901
