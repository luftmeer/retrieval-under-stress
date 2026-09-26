"""Interactive explorer for Retrieval Under Stress."""

import os
from itertools import combinations
from pathlib import Path

import streamlit as st

from api_client import (
    ApiClientError,
    health,
    search,
)
from benchmark import (
    CONDITION_NAMES,
    METHOD_NAMES,
    load_bundle,
    query_ids_for_condition,
    relevant,
    relevant_ranks,
    stress_conditions,
)


HERE = Path(__file__).resolve().parent

API_URL = os.environ.get(
    "RETRIEVAL_API_URL",
    "http://127.0.0.1:8000",
)

BUNDLE_PATH = Path(
    os.environ.get(
        "RETRIEVAL_DEMO_BUNDLE",
        str(HERE / "benchmark-v1.json"),
    )
)


st.set_page_config(
    page_title="Retrieval Under Stress",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_bundle(
    path: str,
    modified_ns: int,
) -> dict:
    del modified_ns

    return load_bundle(
        Path(path)
    )


bundle = cached_bundle(
    str(BUNDLE_PATH),
    BUNDLE_PATH.stat().st_mtime_ns,
)

st.title("Retrieval Under Stress")

st.caption(
    "Explore lexical, dense and hybrid retrieval "
    "and inspect how frozen query perturbations "
    "change their rankings."
)


with st.sidebar:
    st.subheader("System")

    try:
        api_health = health(API_URL)

        st.success("API online")

        st.metric(
            "Documents",
            api_health["documents"],
        )

        st.caption(
            "Dense model: "
            + api_health["model_id"]
        )

    except ApiClientError as exc:
        st.error("API offline")

        st.caption(str(exc))

    st.divider()

    st.subheader("Experiment")

    st.write(
        f"Candidate depth: "
        f"**{bundle['candidate_depth']}**"
    )

    st.write(
        f"RRF constant: "
        f"**{bundle['rrf_k']}**"
    )

    st.write(
        "Metric cutoff: **10**"
    )

    st.caption(
        "The robustness explorer uses frozen "
        "benchmark evidence rather than rerunning "
        "the experiment."
    )
    
    
SCORE_LABELS = {
    "bm25": "BM25 score",
    "dense": "Cosine similarity",
    "hybrid": "RRF score",
}


def render_live_hits(
    method: str,
    hits: list[dict],
) -> None:
    if not hits:
        st.info("No results.")
        return

    for hit in hits:
        with st.container(border=True):
            st.markdown(
                f"**#{hit['rank']} — "
                f"{hit['title'] or 'Untitled'}**"
            )

            st.caption(
                f"Document {hit['document_id']} · "
                f"{SCORE_LABELS[method]}: "
                f"{hit['score']:.5f}"
            )

            st.write(hit["text"])


def ranking_overlap(
    results: dict,
) -> list[dict]:
    rows = []

    for left, right in combinations(
        results,
        2,
    ):
        left_ids = {
            hit["document_id"]
            for hit in results[left]["hits"]
        }

        right_ids = {
            hit["document_id"]
            for hit in results[right]["hits"]
        }

        rows.append(
            {
                "Comparison": (
                    f"{METHOD_NAMES[left]} ↔ "
                    f"{METHOD_NAMES[right]}"
                ),
                "Shared documents": len(
                    left_ids & right_ids
                ),
            }
        )

    return rows



live_tab, robustness_tab = st.tabs(
    [
        "Live Search",
        "Robustness Explorer",
    ]
)

with live_tab:
    st.subheader("Live search")

    st.write(
        "Search arbitrary text against the frozen "
        "SciFact corpus through the FastAPI service."
    )

    with st.form("live-search"):
        query = st.text_input(
            "Query",
            value=(
                "What causes cellular inflammation?"
            ),
        )

        method = st.radio(
            "Retrieval method",
            options=[
                "hybrid",
                "bm25",
                "dense",
            ],
            format_func=lambda value: (
                METHOD_NAMES[value]
            ),
            horizontal=True,
        )

        top_k = st.slider(
            "Displayed top_k",
            min_value=1,
            max_value=20,
            value=10,
        )

        compare_all = st.checkbox(
            "Compare all retrieval methods",
            value=False,
        )

        submitted = st.form_submit_button(
            "Search",
            type="primary",
        )

    if submitted:
        methods = (
            ["bm25", "dense", "hybrid"]
            if compare_all
            else [method]
        )

        results = {}

        try:
            with st.spinner("Searching..."):
                for selected_method in methods:
                    results[selected_method] = search(
                        API_URL,
                        query=query,
                        method=selected_method,
                        top_k=top_k,
                    )

            st.session_state[
                "live_results"
            ] = results

            st.session_state[
                "live_query"
            ] = query

        except ApiClientError as exc:
            st.error(str(exc))

    results = st.session_state.get(
        "live_results"
    )

    if results:
        st.divider()

        st.caption(
            "Raw score magnitudes are comparable "
            "within a method, not across methods."
        )

        if len(results) > 1:
            st.subheader("Ranking overlap")

            st.dataframe(
                ranking_overlap(results),
                hide_index=True,
                use_container_width=True,
            )

        tabs = st.tabs(
            [
                METHOD_NAMES[method]
                for method in results
            ]
        )

        for tab, (
            selected_method,
            result,
        ) in zip(
            tabs,
            results.items(),
        ):
            with tab:
                render_live_hits(
                    selected_method,
                    result["hits"],
                )
                
with robustness_tab:
    st.subheader("Robustness explorer")

    st.write(
        "Inspect queries from the frozen robustness "
        "benchmark. Unlike Live Search, these queries "
        "have relevance judgments and saved metrics."
    )

    conditions = stress_conditions(bundle)

    condition = st.selectbox(
        "Stress condition",
        conditions,
        format_func=lambda value: (
            CONDITION_NAMES[value]
        ),
    )

    available_queries = (
        query_ids_for_condition(
            bundle,
            condition,
        )
    )

    def query_label(query_id: str) -> str:
        text = bundle[
            "queries"
        ][query_id]["variants"]["clean"]

        short = (
            text
            if len(text) <= 90
            else text[:87] + "..."
        )

        return f"{query_id} — {short}"

    query_id = st.selectbox(
        "Benchmark query",
        available_queries,
        format_func=query_label,
    )

    display_k = st.slider(
        "Ranking depth",
        min_value=1,
        max_value=bundle["max_rank"],
        value=10,
        key="benchmark-top-k",
    )

    record = bundle["queries"][query_id]
    
    clean_text = record[
        "variants"
    ]["clean"]

    stressed_text = record[
        "variants"
    ][condition]

    left, right = st.columns(2)

    with left:
        st.markdown("#### Clean")

        st.info(clean_text)

    with right:
        st.markdown(
            f"#### {CONDITION_NAMES[condition]}"
        )

        st.warning(stressed_text)
        
    st.markdown("### Retrieval quality")

    metric_rows = []

    for method in bundle["methods"]:
        clean_metrics = record[
            "metrics"
        ]["clean"][method]

        stressed_metrics = record[
            "metrics"
        ][condition][method]

        metric_rows.append(
            {
                "Method": METHOD_NAMES[method],
                "Clean nDCG@10": round(
                    clean_metrics["ndcg"],
                    3,
                ),
                "Stressed nDCG@10": round(
                    stressed_metrics["ndcg"],
                    3,
                ),
                "nDCG loss": round(
                    clean_metrics["ndcg"]
                    - stressed_metrics["ndcg"],
                    3,
                ),
                "Clean Recall@10": round(
                    clean_metrics["recall"],
                    3,
                ),
                "Stressed Recall@10": round(
                    stressed_metrics["recall"],
                    3,
                ),
                "Clean MRR@10": round(
                    clean_metrics["rr"],
                    3,
                ),
                "Stressed MRR@10": round(
                    stressed_metrics["rr"],
                    3,
                ),
            }
        )

    st.dataframe(
        metric_rows,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Positive nDCG loss means quality "
        "decreased after perturbation."
    )
    
    
    st.markdown(
        "### Ranking changes"
    )

    method_tabs = st.tabs(
        [
            METHOD_NAMES[method]
            for method in bundle["methods"]
        ]
    )

    for method_tab, method in zip(
        method_tabs,
        bundle["methods"],
    ):
        with method_tab:
            clean_hits = record[
                "rankings"
            ]["clean"][method][:display_k]

            stressed_hits = record[
                "rankings"
            ][condition][method][:display_k]

            clean_relevant = relevant_ranks(
                clean_hits,
                record,
            )

            stressed_relevant = relevant_ranks(
                stressed_hits,
                record,
            )

            summary_left, summary_right = (
                st.columns(2)
            )

            with summary_left:
                st.metric(
                    "Relevant ranks — clean",
                    (
                        ", ".join(
                            map(str, clean_relevant)
                        )
                        if clean_relevant
                        else (
                            f"none in top "
                            f"{display_k}"
                        )
                    ),
                )

            with summary_right:
                st.metric(
                    "Relevant ranks — stressed",
                    (
                        ", ".join(
                            map(str, stressed_relevant)
                        )
                        if stressed_relevant
                        else (
                            f"none in top "
                            f"{display_k}"
                        )
                    ),
                )
                
                
            clean_column, stressed_column = (
                st.columns(2)
            )

            def render_saved_ranking(
                column,
                hits,
                heading,
            ):
                with column:
                    st.markdown(
                        f"#### {heading}"
                    )

                    for hit in hits:
                        document_id = hit[
                            "document_id"
                        ]

                        document = bundle[
                            "documents"
                        ][document_id]

                        is_relevant = relevant(
                            record,
                            document_id,
                        )

                        marker = (
                            "✅"
                            if is_relevant
                            else "•"
                        )

                        title = (
                            document["title"]
                            or "Untitled"
                        )

                        with st.container(
                            border=True
                        ):
                            st.markdown(
                                f"{marker} "
                                f"**#{hit['rank']} "
                                f"{title}**"
                            )

                            label = (
                                SCORE_LABELS[method]
                            )

                            st.caption(
                                f"Document "
                                f"{document_id} · "
                                f"{label}: "
                                f"{hit['score']:.5f}"
                            )

                            if is_relevant:
                                grade = record[
                                    "qrels"
                                ][document_id]

                                st.success(
                                    "Judged relevant "
                                    f"(grade {grade})"
                                )

                            st.write(
                                document[
                                    "snippet"
                                ]
                            )

            render_saved_ranking(
                clean_column,
                clean_hits,
                "Clean ranking",
            )

            render_saved_ranking(
                stressed_column,
                stressed_hits,
                CONDITION_NAMES[
                    condition
                ],
            )