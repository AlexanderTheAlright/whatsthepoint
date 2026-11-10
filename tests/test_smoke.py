"""
Smoke tests: call every public function in whatsthepoint.visuals and
whatsthepoint.analyses once on small seeded synthetic data. Each call must
return without error and leave no matplotlib figure open.

Functions that need an optional extra are skipped when it is missing.
"""

import importlib.util
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from whatsthepoint import visuals as V
from whatsthepoint import analyses as A


def has(module):
    return importlib.util.find_spec(module) is not None


rng = np.random.default_rng(42)
N = 120

DF = pd.DataFrame({
    "group": rng.choice(["A", "B", "C"], N),
    "sex": rng.choice(["F", "M"], N),
    "x": rng.normal(0, 1, N),
    "z": rng.normal(0, 1, N),
    "age": rng.integers(20, 65, N),
})
DF["y"] = 1.5 + 0.8 * DF["x"] - 0.4 * DF["z"] + rng.normal(0, 1, N)
DF["d"] = (DF["y"] > DF["y"].median()).astype(int)
DF["count"] = rng.poisson(np.exp(0.3 + 0.2 * DF["x"]))
DF["unit"] = np.repeat(np.arange(N // 4), 4)
DF["time"] = np.tile(np.arange(4), N // 4)
DF["post"] = (DF["time"] >= 2).astype(int)
DF["treat"] = (DF["unit"] % 2).astype(int)

BARS = pd.DataFrame({"category": ["Low", "Mid", "High", "Top"], "value": [12.0, 30.5, 22.25, 8.0]})
GROUPED = pd.DataFrame({"wave": ["2023", "2024", "2025"], "value": [0, 0, 0],
                        "Canada": [40.0, 45.0, 50.0], "US": [35.0, 38.0, 41.0]})
COEFS = pd.DataFrame({
    "term": ["Age", "Education", "Income", "Union"],
    "estimate": [0.12, -0.30, 0.45, 0.05],
    "conf.low": [0.02, -0.50, 0.20, -0.10],
    "conf.high": [0.22, -0.10, 0.70, 0.20],
    "p.value": [0.01, 0.003, 0.0001, 0.4],
})
TIME = pd.DataFrame({
    "date": np.tile(np.arange(30), 2),
    "value": np.r_[np.sin(np.arange(30) / 4), np.cos(np.arange(30) / 4)],
    "category": np.repeat(["a", "b"], 30),
})
COORDS = pd.DataFrame(rng.normal(0, 1, (8, 3)), index=[f"v{i}" for i in range(8)])
TERMS = pd.DataFrame({"terms": [["work", "pay", "union"], ["pay", "boss"], ["union", "work", "strike"],
                                ["boss", "work"], ["pay", "strike", "union"]] * 4})
TREE = pd.DataFrame({"category": ["A", "A", "B", "B"], "subcategory": ["a1", "a2", "b1", "b2"],
                     "value": [10, 20, 15, 5]})
STAGES = pd.DataFrame({"s1": rng.choice(["x", "y"], 40), "s2": rng.choice(["p", "q"], 40),
                       "s3": rng.choice(["m", "n"], 40)})
FREQ = {"work": 30, "pay": 22, "union": 15, "boss": 9, "strike": 5, "hours": 4}
IMPORTANCE = {"age": 0.30, "education": -0.22, "income": 0.15, "union": -0.05}
TEXTS = ["Workers organized a union at the plant.",
         "The union won higher pay for workers.",
         "Managers resisted the strike at the plant.",
         "Pay and hours were the main issues for workers."]

TMP = Path(tempfile.mkdtemp(prefix="whatsthepoint_smoke_"))


def raises(exc, func, *args, **kwargs):
    with pytest.raises(exc):
        func(*args, **kwargs)


OLS = A.regress.ols(DF, "y", ["x", "z"])
LOGIT = A.regress.logit(DF, "d", ["x", "z"])


def rf_model():
    from sklearn.ensemble import RandomForestRegressor
    X = DF[["x", "z", "age"]]
    return RandomForestRegressor(n_estimators=10, random_state=0).fit(X, DF["y"]), X


CASES = {
    # visuals
    "apc.apc_heatmap": lambda: V.apc.apc_heatmap(rng.random((4, 5)), [20, 30, 40, 50], [2000, 2005, 2010, 2015, 2020], list(range(1950, 1958))),
    "apc.apc_heatmap_hex": lambda: V.apc.apc_heatmap(rng.random(20), [20, 30, 40, 50], [2000, 2005, 2010, 2015, 2020], list(range(1950, 1958)), layout="hexagonal"),
    "apc.apc_heatmap_interactive": lambda: V.apc.apc_heatmap_interactive(rng.random((4, 5)), [20, 30, 40, 50], [2000, 2005, 2010, 2015, 2020], list(range(1950, 1958))),
    "bar.bar": lambda: V.bar.bar(BARS, "category", "value", title="Bars"),
    "bar.bar_horizontal": lambda: V.bar.bar(BARS, "category", "value", orientation="horizontal"),
    "bar.bar_grouped": lambda: V.bar.bar(GROUPED, "wave", "value", orientation="grouped"),
    "bar.bar_stacked": lambda: V.bar.bar(GROUPED, "wave", "value", orientation="stacked"),
    "bar.bar_reviewer3": lambda: V.bar.bar(BARS, "category", "value", style_mode="reviewer3"),
    "bar.bar_subplots": lambda: V.bar.bar_subplots(BARS.assign(v2=BARS["value"] * 2), "category", ["value", "v2"]),
    "bar.bar_interactive": lambda: V.bar.bar_interactive(BARS, "category", "value"),
    "boxplot.boxplot": lambda: V.boxplot.boxplot(DF, x="group", y="y"),
    "boxplot.boxplot_subplots": lambda: V.boxplot.boxplot_subplots(DF, "group", ["y", "x"]),
    "boxplot.boxplot_interactive": lambda: V.boxplot.boxplot_interactive(DF, x="group", y="y"),
    "cluster.cluster": lambda: V.cluster.cluster(DF[["x", "z", "y"]].head(20)),
    "cluster.cluster_interactive": lambda: V.cluster.cluster_interactive(DF[["x", "z", "y"]].head(20)),
    "cluster.heatmap_cluster": lambda: V.cluster.heatmap_cluster(DF[["x", "z", "y"]].head(15)),
    "coef.coef": lambda: V.coef.coef(COEFS, p_col="p.value"),
    "coef.coef_interactive": lambda: V.coef.coef_interactive(COEFS, p_col="p.value"),
    "coef.sigstars": lambda: V.coef.sigstars(0.01),
    "cooccur.cooccur": lambda: V.cooccur.cooccur(TERMS, "terms", top_n=2),
    "cooccur.cooccur_interactive": lambda: V.cooccur.cooccur_interactive(TERMS, "terms", top_n=2),
    "dag.dag": lambda: V.dag.dag(["X", "M", "Y"], [("X", "M"), ("M", "Y"), ("X", "Y")]),
    "dag.dag_from_formula": lambda: V.dag.dag_from_formula("Y ~ X + M"),
    "dag.dag_interactive": lambda: V.dag.dag_interactive(["X", "M", "Y"], [("X", "M"), ("M", "Y")]),
    "density.kde": lambda: V.density.kde(DF, "y", group="group"),
    "density.kde_interactive": lambda: V.density.kde_interactive(DF, "y", group="group"),
    "density.raincloud": lambda: V.density.raincloud(DF, "group", "y"),
    "density.ridgeline": lambda: V.density.ridgeline(DF, "y", "group"),
    "density.ryder": lambda: V.density.ryder(DF, "y", "group"),
    "density.ryder_interactive": lambda: V.density.ryder_interactive(DF, "y", "group"),
    "factormap.factormap": lambda: V.factormap.factormap(COORDS[[0, 1]], sup_coords=COORDS[[0, 1]].head(3) * 0.5),
    "factormap.factormap_interactive": lambda: V.factormap.factormap_interactive(COORDS[[0, 1]]),
    "factormap.factormap_3d": lambda: V.factormap.factormap_3d(COORDS),
    "factormap.factormap_3d_interactive": lambda: V.factormap.factormap_3d_interactive(COORDS),
    "factormap.variance_explained": lambda: V.factormap.variance_explained([0.4, 0.25, 0.15, 0.1, 0.1]),
    "factormap.variance_explained_interactive": lambda: V.factormap.variance_explained_interactive([0.4, 0.25, 0.15, 0.1, 0.1]),
    "feature.feature_importance_bar": lambda: V.feature.feature_importance_bar(IMPORTANCE),
    "feature.feature_importance_bar_interactive": lambda: V.feature.feature_importance_bar_interactive(IMPORTANCE),
    "feature.feature_waterfall": lambda: V.feature.feature_waterfall(IMPORTANCE, base_value=0.5),
    "feature.feature_waterfall_interactive": lambda: V.feature.feature_waterfall_interactive(IMPORTANCE, base_value=0.5),
    "flow.sankey": lambda: V.flow.sankey([0, 0, 1, 1, 2], [2, 3, 2, 3, 4], [10, 15, 20, 5, 25], ["A", "B", "C", "D", "E"]),
    "flow.sankey_interactive": lambda: V.flow.sankey_interactive([0, 0, 1], [1, 2, 2], [5, 3, 2], ["A", "B", "C"]),
    "flow.alluvial": lambda: V.flow.alluvial(STAGES, ["s1", "s2", "s3"]),
    "flow.alluvial_interactive": lambda: V.flow.alluvial_interactive(STAGES, ["s1", "s2", "s3"]),
    "geographic.point_map": lambda: V.geographic.point_map(DF.assign(lat=DF["x"] + 45, lon=DF["z"] - 75), "lat", "lon", value="y"),
    "geographic.point_map_interactive": lambda: V.geographic.point_map_interactive(DF.assign(lat=DF["x"] + 45, lon=DF["z"] - 75), "lat", "lon"),
    "geographic.choropleth": lambda: V.geographic.choropleth(["CA", "NY", "TX"], [1, 2, 3]),
    "geographic.choropleth_interactive": lambda: V.geographic.choropleth_interactive(["CA", "NY", "TX"], [1, 2, 3]),
    "geographic.hexbin_map": lambda: V.geographic.hexbin_map(DF.assign(lat=DF["x"] + 45, lon=DF["z"] - 75), "lat", "lon"),
    "heatmap.heatmap": lambda: V.heatmap.heatmap(DF[["x", "z", "y"]]),
    "heatmap.heatmap_interactive": lambda: V.heatmap.heatmap_interactive(DF[["x", "z", "y"]]),
    "heatmap.heatmap_distribution": lambda: V.heatmap.heatmap_distribution(pd.DataFrame(rng.random((3, 4)))),
    "heatmap.heatmap_distribution_interactive": lambda: V.heatmap.heatmap_distribution_interactive(pd.DataFrame(rng.random((3, 4)))),
    "hierarchical.treemap": lambda: V.hierarchical.treemap(TREE, ["category", "subcategory"], "value"),
    "hierarchical.treemap_interactive": lambda: V.hierarchical.treemap_interactive(TREE, ["category", "subcategory"], "value"),
    "hierarchical.sunburst": lambda: V.hierarchical.sunburst(TREE, ["category", "subcategory"], "value"),
    "hierarchical.sunburst_interactive": lambda: V.hierarchical.sunburst_interactive(TREE, ["category", "subcategory"], "value"),
    "hist.histogram": lambda: V.hist.histogram(DF, "y"),
    "hist.histogram_color_by": lambda: V.hist.histogram(DF, "y", color_by="sex"),
    "hist.histogram_interactive": lambda: V.hist.histogram_interactive(DF, "y"),
    "horizon.horizon": lambda: V.horizon.horizon(TIME[TIME.category == "a"], "date", "value"),
    "horizon.horizon_panel": lambda: V.horizon.horizon_panel(TIME, "date", "value", "category"),
    "horizon.horizon_interactive": lambda: V.horizon.horizon_interactive(TIME, "date", "value", group="category"),
    "ice.ice": lambda: V.ice.ice(*rf_model(), "x", n_points=10, sample_size=20),
    "ice.ice_comparison": lambda: V.ice.ice_comparison(*rf_model(), ["x", "z"], n_points=10, sample_size=20),
    "ice.ice_interactive": lambda: V.ice.ice_interactive(*rf_model(), "x", n_points=10, sample_size=20),
    "margins.margins": lambda: V.margins.margins(OLS, "x", n_points=10),
    "margins.margins_comparison": lambda: V.margins.margins_comparison(A.regress.ols(DF, "y", ["x", "z", "d"]), "x", "d", n_points=10),
    "margins.margins_interactive": lambda: V.margins.margins_interactive(OLS, "x", n_points=10),
    "margins.margins_comparison_interactive": lambda: V.margins.margins_comparison_interactive(A.regress.ols(DF, "y", ["x", "z", "d"]), "x", "d", n_points=10),
    "oddsratio.oddsratio": lambda: V.oddsratio.oddsratio(LOGIT),
    "oddsratio.oddsratio_comparison": lambda: V.oddsratio.oddsratio_comparison([LOGIT, A.regress.logit(DF, "d", ["x"])], ["Full", "Reduced"]),
    "oddsratio.oddsratio_interactive": lambda: V.oddsratio.oddsratio_interactive(LOGIT),
    "pair.pair": lambda: V.pair.pair(DF[["x", "z", "y", "group"]], hue="group"),
    "pair.pair_interactive": lambda: V.pair.pair_interactive(DF[["x", "z", "y", "group"]], hue="group"),
    "pie.pie": lambda: V.pie.pie(DF, "group"),
    "pie.pie_interactive": lambda: V.pie.pie_interactive(DF, "group"),
    "residuals.residuals": lambda: V.residuals.residuals(OLS),
    "residuals.residuals_histogram": lambda: V.residuals.residuals_histogram(OLS),
    "residuals.residuals_interactive": lambda: V.residuals.residuals_interactive(OLS),
    "scatter.scatterplot": lambda: V.scatter.scatterplot(DF, "x", "y", group="sex"),
    "scatter.scatterplot_interactive": lambda: V.scatter.scatterplot_interactive(DF, "x", "y", group="sex"),
    "trend.trend": lambda: V.trend.trend(TIME, "date", "value", group="category"),
    "trend.trend_bar": lambda: V.trend.trend(TIME[TIME.category == "a"].assign(value=lambda d: d.value + 2), "date", "value", kind="bar"),
    "trend.trend_interactive": lambda: V.trend.trend_interactive(TIME, "date", "value", group="category"),
    "waterfall.waterfall": lambda: V.waterfall.waterfall(BARS, "category", "value"),
    "waterfall.waterfall_interactive": lambda: V.waterfall.waterfall_interactive(BARS, "category", "value"),
    "wordcloud.wordcloud": lambda: V.wordcloud.wordcloud(FREQ),
    "wordcloud.wordcloud_interactive": lambda: V.wordcloud.wordcloud_interactive(FREQ, exact_words=6),
    "density.ryder_top": lambda: V.ryder(DF, "y", "group"),
    # analyses: regress
    "regress.ols": lambda: OLS.get_tidy() if hasattr(OLS, "get_tidy") else OLS,
    "regress.logit": lambda: A.regress.logit(DF, "d", ["x", "z"]),
    "regress.poisson": lambda: A.regress.poisson(DF, "count", ["x"]),
    "regress.regress": lambda: A.regress.regress("y ~ x + z", DF),
    "regress.compare_models": lambda: A.regress.compare_models(OLS, A.regress.ols(DF, "y", ["x"])),
    "regress.marginal_effects": lambda: A.regress.marginal_effects(LOGIT),
    "regress.multilevel": lambda: A.regress.multilevel(DF, "y", ["x"], groups="unit").get_tidy(),
    "regress.RegressionModel": lambda: A.regress.RegressionModel(DF, "y", ["x"]),
    "regress.MultilevelModel": lambda: A.regress.MultilevelModel(DF, "y", ["x"], groups="unit"),
    # analyses: sem
    "sem.path_analysis": lambda: A.sem.path_analysis(DF, {"z": ["x"], "y": ["x", "z"]}),
    "sem.PathModel": lambda: A.sem.PathModel(DF, {"y": ["x", "z"]}),
    "sem.mediation": lambda: A.sem.mediation(DF, "x", "z", "y"),
    "sem.cfa": lambda: raises(NotImplementedError, A.sem.cfa, DF, {"F": ["x", "z", "y"]}),
    "sem.lavaan_interface": lambda: raises(NotImplementedError, A.sem.lavaan_interface, DF, "F =~ x + z + y"),
    # analyses: model_io and pubtable
    "model_io.save_model": lambda: A.model_io.save_model(OLS, TMP / "m.json"),
    "model_io.save_models": lambda: A.model_io.save_models([OLS, LOGIT], TMP / "ms.json", model_names=["OLS", "Logit"]),
    "model_io.load_model": lambda: (A.model_io.save_model(OLS, TMP / "m2.json"), A.model_io.load_model(TMP / "m2.json")),
    "model_io.load_models": lambda: (A.model_io.save_models([OLS, LOGIT], TMP / "ms2.json"), A.model_io.load_models(TMP / "ms2.json")),
    "model_io.models_to_dataframes": lambda: (A.model_io.save_models([OLS, LOGIT], TMP / "ms3.json"), A.model_io.models_to_dataframes(A.model_io.load_models(TMP / "ms3.json"))),
    "model_io.export_for_r": lambda: A.model_io.export_for_r(OLS, TMP / "for_r", include_vcov=True),
    "model_io.import_from_r": lambda: (A.model_io.export_for_r(OLS, TMP / "for_r2"), A.model_io.import_from_r(TMP / "for_r2_coef.csv")),
    "pubtable.descriptive_table": lambda: A.pubtable.descriptive_table(DF, ["x", "y"], group_var="sex"),
    "pubtable.proportion_table": lambda: A.pubtable.proportion_table(DF, "group", "sex"),
    "pubtable.regression_table": lambda: A.pubtable.regression_table([OLS.get_tidy(), A.regress.ols(DF, "y", ["x"]).get_tidy()], ["M1", "M2"]),
    "pubtable.save_table": lambda: A.pubtable.save_table("<table></table>", TMP / "t.html"),
    # analyses: descriptive
    "descriptive.compare_groups": lambda: A.descriptive.compare_groups(DF, "y", "sex"),
    "descriptive.correlation_matrix": lambda: A.descriptive.correlation_matrix(DF, ["x", "z", "y"]),
    "descriptive.crosstab": lambda: A.descriptive.crosstab(DF, "group", "sex"),
    "descriptive.describe_by_group": lambda: A.descriptive.describe_by_group(DF, ["x", "y"], "group"),
    "descriptive.distribution_test": lambda: A.descriptive.distribution_test(DF, "y"),
    "descriptive.group_summary": lambda: A.descriptive.group_summary(DF, ["x", "y"], "group"),
    # analyses: causal
    "causal.difference_in_differences": lambda: A.causal.difference_in_differences(DF, "y", "treat", "post", "unit"),
    "causal.instrumental_variables": lambda: A.causal.instrumental_variables(DF.assign(endo=DF.x + DF.z), "y", "endo", "z", exogenous="age"),
    "causal.propensity_score": lambda: A.causal.propensity_score(DF, "treat", ["x", "z"], outcome="y"),
    "causal.PropensityScoreAnalysis": lambda: A.causal.PropensityScoreAnalysis(DF, "treat", ["x", "z"], outcome="y"),
    "causal.regression_discontinuity": lambda: A.causal.regression_discontinuity(DF, "y", "x", 0.0),
    # analyses: panel
    "panel.fixed_effects": lambda: A.panel.fixed_effects(DF, "y", ["x"], "unit", "time"),
    "panel.first_differences": lambda: A.panel.first_differences(DF, "y", ["x"], "unit", "time"),
    "panel.panel_summary": lambda: A.panel.panel_summary(DF, ["x", "y"], "unit", "time"),
    "panel.random_effects": pytest.param(lambda: A.panel.random_effects(DF, "y", ["x"], "unit", "time"),
                                         marks=pytest.mark.skipif(not has("linearmodels"), reason="needs whatsthepoint[panel]")),
    "panel.hausman_test": pytest.param(lambda: A.panel.hausman_test(A.panel.fixed_effects(DF, "y", ["x"], "unit", "time"), A.panel.random_effects(DF, "y", ["x"], "unit", "time")),
                                       marks=pytest.mark.skipif(not has("linearmodels"), reason="needs whatsthepoint[panel]")),
    "panel.compare_panel_models": pytest.param(lambda: A.panel.compare_panel_models(A.panel.fixed_effects(DF, "y", ["x"], "unit", "time"), A.panel.random_effects(DF, "y", ["x"], "unit", "time")),
                                               marks=pytest.mark.skipif(not has("linearmodels"), reason="needs whatsthepoint[panel]")),
    # analyses: ml
    "ml.build_pipeline": lambda: A.ml.build_pipeline(["x", "z"], ["group"]),
    "ml.train_model": lambda: A.ml.train_model(DF, "d", ["x", "z", "group"]),
    "ml.MLModel": lambda: A.ml.MLModel(DF, "y", ["x", "z"], task="regression"),
    "ml.feature_importance": lambda: A.ml.feature_importance(A.ml.train_model(DF, "d", ["x", "z"])),
    "ml.shap_analysis": pytest.param(lambda: A.ml.shap_analysis(A.ml.train_model(DF, "d", ["x", "z"]), DF[["x", "z"]].head(20)),
                                     marks=pytest.mark.skipif(not has("shap"), reason="needs whatsthepoint[ml]")),
    # analyses: text
    "text.analyze_corpus": lambda: A.text_analysis.analyze_corpus(TEXTS, n_topics=2),
    "text.average_word_length": lambda: A.text_analysis.average_word_length(TEXTS[0]),
    "text.clean_text": lambda: A.text_analysis.clean_text(TEXTS[0]),
    "text.complexity_scores": lambda: A.text_analysis.complexity_scores(TEXTS[1]),
    "text.cosine_similarity_texts": lambda: A.text_analysis.cosine_similarity_texts(TEXTS),
    "text.create_tfidf_matrix": lambda: A.text_analysis.create_tfidf_matrix(TEXTS),
    "text.extract_ngrams": lambda: A.text_analysis.extract_ngrams(TEXTS[0]),
    "text.flesch_kincaid_grade": lambda: A.text_analysis.flesch_kincaid_grade(TEXTS[0]),
    "text.flesch_reading_ease": lambda: A.text_analysis.flesch_reading_ease(TEXTS[0]),
    "text.get_top_tfidf_terms": lambda: A.text_analysis.get_top_tfidf_terms(A.text_analysis.create_tfidf_matrix(TEXTS)[0]),
    "text.jaccard_similarity": lambda: A.text_analysis.jaccard_similarity(TEXTS[0], TEXTS[1]),
    "text.levenshtein_distance": lambda: A.text_analysis.levenshtein_distance("union", "unions"),
    "text.lexical_diversity": lambda: A.text_analysis.lexical_diversity(TEXTS[0]),
    "text.ngram_frequency": lambda: A.text_analysis.ngram_frequency(TEXTS),
    "text.remove_stopwords": lambda: A.text_analysis.remove_stopwords(["the", "union", "won"]),
    "text.tokenize": lambda: A.text_analysis.tokenize(TEXTS[0]),
    "text.TopicModel": lambda: A.text_analysis.TopicModel(n_topics=2, random_state=0).fit(TEXTS),
    "text.analyze_sentiment": pytest.param(lambda: A.text_analysis.analyze_sentiment(TEXTS[0]),
                                           marks=pytest.mark.skip(reason="downloads a transformer model")),
    "text.bert_embeddings": pytest.param(lambda: A.text_analysis.bert_embeddings(TEXTS[0]),
                                         marks=pytest.mark.skip(reason="downloads a transformer model")),
    "text.BERTModel": pytest.param(lambda: A.text_analysis.BERTModel(),
                                   marks=pytest.mark.skip(reason="downloads a transformer model")),
    "text.SentimentAnalyzer": pytest.param(lambda: A.text_analysis.SentimentAnalyzer(),
                                           marks=pytest.mark.skip(reason="downloads a transformer model")),
    "text.OllamaClassifier": pytest.param(lambda: A.text_analysis.OllamaClassifier(),
                                          marks=pytest.mark.skipif(not has("requests"), reason="needs requests")),
    # analyses: network
    "network.adjacency_to_edgelist": lambda: A.network.adjacency_to_edgelist(pd.DataFrame([[0, 1], [1, 0]], index=["a", "b"], columns=["a", "b"])),
    "network.cooccurrence_network": lambda: A.network.cooccurrence_network(TERMS.explode("terms").reset_index(), "terms", "index"),
    "network.correlation_network": lambda: A.network.correlation_network(DF[["x", "z", "y"]], threshold=0.1),
    "network.create_adjacency_matrix": lambda: A.network.create_adjacency_matrix(A.network.edgelist_from_pairs([("a", "b"), ("b", "c")])),
    "network.create_bipartite_edgelist": lambda: A.network.create_bipartite_edgelist(DF.head(20), "group", "sex"),
    "network.create_edgelist": lambda: A.network.create_edgelist(DF.head(20), "group", "sex"),
    "network.edgelist_from_pairs": lambda: A.network.edgelist_from_pairs([("a", "b"), ("b", "c")]),
    "network.network_summary": lambda: A.network.network_summary(A.network.edgelist_from_pairs([("a", "b"), ("b", "c")])),
    "network.project_bipartite": lambda: A.network.project_bipartite(A.network.create_bipartite_edgelist(DF.head(20), "group", "sex")),
    "network.similarity_network": lambda: A.network.similarity_network(DF.head(10).assign(id=range(10)), ["x", "z"], "id", threshold=0.0),
    "network.transition_network": lambda: A.network.transition_network([["a", "b", "c"], ["a", "c"]]),
    "network.word_cooccurrence_network": lambda: A.network.word_cooccurrence_network(TEXTS, min_cooccurrence=1),
}

NEEDS = {
    "cooccur.": "networkx",
    "dag.": "networkx",
    "factormap.factormap": "adjustText",
    "wordcloud.wordcloud": "wordcloud",
}


def _params():
    for name, case in CASES.items():
        if isinstance(case, type(pytest.param(None))):
            yield pytest.param(*case.values, marks=case.marks, id=name)
            continue
        mod = next((m for prefix, m in NEEDS.items() if name.startswith(prefix)), None)
        if mod and not has(mod):
            yield pytest.param(case, marks=pytest.mark.skip(reason=f"needs {mod}"), id=name)
        else:
            yield pytest.param(case, id=name)


@pytest.mark.parametrize("call", list(_params()))
def test_public_function_runs(call):
    plt.close("all")
    call()
    plt.close("all")
    assert plt.get_fignums() == []


def test_every_public_function_has_a_case():
    import inspect
    covered = {k.split("_reviewer3")[0] for k in CASES}
    missing = []
    for pkg, prefix_map in ((V, {}), (A, {"text_analysis": "text"})):
        for mname in pkg.__all__:
            mod = getattr(pkg, mname)
            if not inspect.ismodule(mod):
                continue
            key = prefix_map.get(mname, mname)
            for n, obj in inspect.getmembers(mod, lambda o: inspect.isfunction(o) or inspect.isclass(o)):
                if n.startswith("_") or obj.__module__ != mod.__name__:
                    continue
                if mname == "apc" and n in ("get_color", "get_data_element_kwargs"):
                    continue
                if f"{key}.{n}" not in CASES:
                    missing.append(f"{key}.{n}")
    assert missing == [], missing


def test_save_figure_writes_svg_and_png(tmp_path):
    from whatsthepoint.utils.style import save_figure
    fig, _ = V.bar.bar(BARS, "category", "value")
    svg, png = save_figure(fig, tmp_path / "fig.01.bars")
    plt.close("all")
    assert svg.exists() and png.exists()


def test_bars_start_at_zero():
    fig, ax = V.bar.bar(BARS.assign(value=BARS["value"] + 50), "category", "value")
    assert ax.get_ylim()[0] == 0
    plt.close("all")


def test_reviewer3_two_groups_black_and_white():
    from whatsthepoint.utils.style import generate_semantic_palette
    pal = generate_semantic_palette({"positive": ["a", "b"]}, mode="reviewer3")
    assert pal["a"][:3] == (0.0, 0.0, 0.0)
    assert pal["b"][:3] == (1.0, 1.0, 1.0)


def test_default_style_mode_is_house():
    import inspect
    for pkg in (V,):
        for mname in pkg.__all__:
            mod = getattr(pkg, mname)
            if not inspect.ismodule(mod):
                continue
            for n, f in inspect.getmembers(mod, inspect.isfunction):
                if n.startswith("_") or f.__module__ != mod.__name__:
                    continue
                p = inspect.signature(f).parameters.get("style_mode")
                if p is not None and p.default is not inspect._empty:
                    assert p.default == "house", f"{mname}.{n}"
