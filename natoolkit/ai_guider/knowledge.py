from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .router import RouteDecision


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MAX_BLOCK_CHARS = 18_000
MAX_CONTEXT_CHARS = 52_000


class KnowledgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SymbolRef:
    path: str
    symbol: str


@dataclass(frozen=True)
class TopicSpec:
    description: str
    documents: tuple[str, ...]
    symbols: tuple[SymbolRef, ...] = ()


@dataclass(frozen=True)
class SourceBlock:
    source_id: str
    label: str
    content: str


@dataclass(frozen=True)
class ContextBundle:
    sources: tuple[SourceBlock, ...]

    def render(self) -> str:
        return "\n\n".join(
            f"<source id=\"{source.source_id}\" label=\"{source.label}\">\n"
            f"{source.content}\n</source>"
            for source in self.sources
        )

    @property
    def source_map(self) -> dict[str, SourceBlock]:
        return {source.source_id: source for source in self.sources}


def _ref(path: str, *symbols: str) -> tuple[SymbolRef, ...]:
    return tuple(SymbolRef(path, symbol) for symbol in symbols)


ACTIVITY_DOC = ("tools/activity_tracer/README.md",)
SLEEP_DOC = ("tools/sleep_state_staging/README.md",)
ALIGNER_DOC = ("tools/activity_sleep_state_aligner/README.md",)
GUIDER_DOC = ("ai_guider/README.md",)


TOPICS: dict[str, TopicSpec] = {
    "project.overview": TopicSpec(
        "Toolkit purpose, applications, installation, and end-to-end workflow.",
        GUIDER_DOC,
    ),
    "launcher.usage": TopicSpec(
        "Unified launcher behavior and application entry points.",
        GUIDER_DOC,
        _ref("launcher/gui.py", "start_tool", "LauncherWindow"),
    ),
    "guider.usage": TopicSpec(
        "AI Guider setup, scope, privacy, and user interaction.",
        GUIDER_DOC,
        _ref("ai_guider/router.py", "parse_route", "RouteDecision"),
    ),
    "activity.workflow": TopicSpec(
        "Activity Tracer launch, movie-to-trace workflow, and controls.",
        ACTIVITY_DOC,
        _ref(
            "tools/activity_tracer/gui.py",
            "SimpleTracerWidget.import_movies",
            "SimpleTracerWidget.extract_all_movies",
        ),
    ),
    "activity.movies": TopicSpec(
        "Movie import, TimeROI, spatial/temporal alignment, and merged view.",
        ACTIVITY_DOC,
        _ref(
            "tools/activity_tracer/gui.py",
            "SimpleTracerWidget.apply_time_roi",
            "SimpleTracerWidget.spatially_align_selected_movies",
            "SimpleTracerWidget.temporally_align_selected_movies",
            "SimpleTracerWidget.show_merged_view",
        ),
    ),
    "activity.roi": TopicSpec(
        "Shared/unique ROI labels, manual editing, copying, and trace extraction.",
        ACTIVITY_DOC,
        _ref("tools/activity_tracer/roi.py", "ROILabels", "roi_centers", "roi_ids")
        + _ref(
            "tools/activity_tracer/gui.py",
            "SimpleTracerWidget.roi_set_for",
            "SimpleTracerWidget.trace_state",
        ),
    ),
    "activity.suite2p": TopicSpec(
        "Optional Suite2p motion correction and automatic ROI detection.",
        ACTIVITY_DOC,
        _ref(
            "tools/activity_tracer/suite2p_adapter.py",
            "run_motion_correction",
            "run_roi_detection",
            "run_roi_detection_on_movie",
        ),
    ),
    "activity.normalization": TopicSpec(
        "Raw traces, dF/F0, SNR, Z-score, min-max, and FRAME event detection.",
        ACTIVITY_DOC,
        _ref(
            "tools/activity_tracer/processing.py",
            "process_frame_trace",
            "find_frame_spikes",
            "normalize_trace",
        ),
    ),
    "activity.output": TopicSpec(
        "ROI, activity CSV, and event CSV export schemas.",
        ACTIVITY_DOC,
        _ref(
            "tools/activity_tracer/gui.py",
            "SimpleTracerWidget.export_roi_labels",
            "SimpleTracerWidget.export_activities",
            "SimpleTracerWidget.export_spikes",
        ),
    ),
    "sleep.workflow": TopicSpec(
        "Sleep State Staging launch, experiment workflow, GUI, and CLI.",
        SLEEP_DOC,
        _ref(
            "tools/sleep_state_staging/gui.py",
            "AnalysisWorker.run",
            "SleepStateReviewWindow.run_analysis",
        ),
    ),
    "sleep.discovery": TopicSpec(
        "Note syntax, EEG/EMG assignment, TIFF matching, and experiment discovery.",
        SLEEP_DOC,
        _ref(
            "tools/sleep_state_staging/experiment.py",
            "scan_experiment",
            "parse_note",
            "load_video_frame_times",
        ),
    ),
    "sleep.preprocessing": TopicSpec(
        "EEG/EMG file loading, channel conventions, high-pass, and notch filtering.",
        SLEEP_DOC,
        _ref(
            "tools/sleep_state_staging/io.py", "probe_eegemg_txt", "load_eegemg_txt"
        )
        + _ref(
            "tools/sleep_state_staging/preprocess.py",
            "remove_dc_offset",
            "remove_power_interference",
            "preprocess_eeg_emg",
        ),
    ),
    "sleep.classification": TopicSpec(
        "EEG/EMG features, Wake score, REM/NREM threshold, and post-processing.",
        SLEEP_DOC,
        _ref(
            "tools/sleep_state_staging/staging.py",
            "StagingParams",
            "classify_sleep_state",
            "_wake_score",
            "_postprocess_labels",
        ),
    ),
    "sleep.review": TopicSpec(
        "Draft edits, video confirmation, undo/redo, commit, and global overview.",
        SLEEP_DOC,
        _ref("tools/sleep_state_staging/review.py", "ReviewSession")
        + _ref(
            "tools/sleep_state_staging/gui.py",
            "LocalReviewPage",
            "GlobalOverviewPage",
        ),
    ),
    "sleep.output": TopicSpec(
        "Automatic/corrected files, summaries, hypnograms, and alignment helpers.",
        SLEEP_DOC,
        _ref(
            "tools/sleep_state_staging/review.py",
            "write_auto_results",
            "_write_corrected_csv",
        )
        + _ref("tools/sleep_state_staging/qc.py", "plot_hypnogram"),
    ),
    "aligner.workflow": TopicSpec(
        "Aligner launch, required inputs, execution, and output.",
        ALIGNER_DOC,
        _ref("tools/activity_sleep_state_aligner/core.py", "align_activity_file"),
    ),
    "aligner.timing": TopicSpec(
        "Note start times, TIFF frame timestamps, interval lookup, and Unknown labels.",
        ALIGNER_DOC,
        _ref(
            "tools/activity_sleep_state_aligner/core.py",
            "align_activity_file",
            "load_note_times",
            "load_sleep_intervals",
            "SleepIntervals",
        ),
    ),
    "aligner.validation": TopicSpec(
        "Input schemas, filename matching, validation errors, and limitations.",
        ALIGNER_DOC,
        _ref(
            "tools/activity_sleep_state_aligner/core.py",
            "movie_key",
            "_load_activity_rows",
            "_load_frame_timestamps",
        ),
    ),
}


def topic_catalog() -> str:
    return "\n".join(
        f"- {name}: {spec.description}" for name, spec in TOPICS.items()
    )


def build_context(route: RouteDecision) -> ContextBundle:
    if not route.in_scope:
        raise KnowledgeError("No project context is available for an out-of-scope question.")

    document_blocks: list[tuple[str, str]] = []
    source_blocks: list[tuple[int, str, str]] = []
    seen_documents: set[str] = set()
    seen_symbols: set[SymbolRef] = set()
    for topic in route.topics:
        spec = TOPICS[topic]
        for relative in spec.documents:
            if relative in seen_documents:
                continue
            seen_documents.add(relative)
            content = _read_package_file(relative)
            if content:
                document_blocks.append((relative, content))
        if route.needs_source:
            for reference in spec.symbols:
                if reference in seen_symbols:
                    continue
                seen_symbols.add(reference)
                extracted = _extract_symbol(reference)
                if extracted is not None:
                    label, content = extracted
                    source_blocks.append(
                        (_term_score(f"{label}\n{content}", route.search_terms), label, content)
                    )

    sources: list[SourceBlock] = []
    used = 0
    blocks = document_blocks + [
        (label, content)
        for _score, label, content in sorted(source_blocks, reverse=True)
    ]
    for label, content in blocks:
        remaining = MAX_CONTEXT_CHARS - used
        if remaining <= 0:
            break
        text = content[: min(MAX_BLOCK_CHARS, remaining)].rstrip()
        if not text:
            continue
        sources.append(SourceBlock(f"S{len(sources) + 1}", label, text))
        used += len(text)
    if not sources:
        raise KnowledgeError("No approved project documentation or source matched the question.")
    return ContextBundle(tuple(sources))


def _term_score(text: str, terms: tuple[str, ...]) -> int:
    content = text.casefold()
    return sum(content.count(term.casefold()) for term in terms if term)


_CITATION_RE = re.compile(r"\[(S\d+)\]")


def ground_answer(answer: str, context: ContextBundle) -> str:
    source_map = context.source_map
    valid_ids: list[str] = []

    def replace(match: re.Match[str]) -> str:
        source_id = match.group(1)
        if source_id not in source_map:
            return ""
        if source_id not in valid_ids:
            valid_ids.append(source_id)
        return match.group(0)

    cleaned = _CITATION_RE.sub(replace, answer or "").strip()
    if not valid_ids:
        return (
            "I could not produce an answer with a verifiable citation from the "
            "approved project documentation or source code."
        )
    sources = "\n".join(
        f"- [{source_id}] {source_map[source_id].label}" for source_id in valid_ids
    )
    return f"{cleaned}\n\nSources\n{sources}"


def _read_package_file(relative: str) -> str:
    path = _approved_path(relative)
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_symbol(reference: SymbolRef) -> tuple[str, str] | None:
    path = _approved_path(reference.path)
    if not path.is_file() or path.suffix != ".py":
        return None
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    node = _find_symbol(tree, reference.symbol.split("."))
    if node is None or node.end_lineno is None:
        return None
    start = min(
        [node.lineno, *(decorator.lineno for decorator in getattr(node, "decorator_list", []))]
    )
    content = "\n".join(text.splitlines()[start - 1 : node.end_lineno])
    return f"{reference.path}:{start} ({reference.symbol})", content


def _find_symbol(tree: ast.AST, parts: list[str]):
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == parts[0]
    ]
    if not candidates:
        return None
    node = candidates[0]
    for name in parts[1:]:
        node = next(
            (
                child
                for child in getattr(node, "body", [])
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == name
            ),
            None,
        )
        if node is None:
            return None
    return node


def _approved_path(relative: str) -> Path:
    path = (PACKAGE_ROOT / relative).resolve()
    if not path.is_relative_to(PACKAGE_ROOT):
        raise KnowledgeError(f"Project context path is outside the package: {relative}")
    return path
