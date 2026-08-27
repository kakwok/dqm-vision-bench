"""Driving logic behind write_truth.ipynb.

The notebook only imports this module, constructs a `TruthWriter`, and calls
its methods / edits its `quote`/`describe`/`compare`/`decision` attributes --
all the path/lookup/rendering logic lives here so the notebook stays a thin
driver.
"""

import base64
import json
import re
import textwrap
from pathlib import Path

import pandas as pd
import yaml
from IPython.display import HTML, display

from owui_client import find_reference_images

TRUTH_ROOT = Path('truth')
IMAGE_ROOT = Path('images')
BAD_IMAGE_ROOT = Path('bad_images')
REF_DIR = Path('ref_images')
PLOT_INSTRUCTIONS_DIR = Path('plot_instructions')
EVENT_TYPE_CONFIG = Path('results/YAML/config_YAML.json')


def load_instructions(plot_instructions_dir: Path = PLOT_INSTRUCTIONS_DIR) -> dict:
    """{plot folder name -> instruction text}, flattened from plot_instructions/*.yaml."""
    instructions = {}
    for yaml_path in sorted(plot_instructions_dir.glob('*.yaml')):
        doc = yaml.safe_load(yaml_path.read_text())
        for spec in doc.get('plots', {}).values():
            instructions[spec['folder']] = spec['instruction']
    return instructions


def load_event_types(config_path: Path = EVENT_TYPE_CONFIG) -> dict:
    """{run number -> event type} sourced from an existing batch run's config."""
    if not config_path.exists():
        return {}
    cfg = json.loads(config_path.read_text())
    return {
        int(k): v
        for k, v in cfg.get('run_metadata', {}).get('event_type_map', {}).items()
    }


def truth_path_for(plot: str, image_path: Path, truth_root: Path = TRUTH_ROOT) -> Path:
    """Mirrors find_truth_file()'s naming rule (evaluate.py), but returns the
    path whether or not it exists yet -- that's what lets save() know where
    to write."""
    stem = image_path.stem
    match = re.search(r'run\d+', stem)
    run = match.group() if match else stem

    if '_bad_small' in stem:
        return truth_root / plot / f'truth_{run}_bad_small.txt'
    elif '_bad_medium' in stem:
        return truth_root / plot / f'truth_{run}_bad_medium.txt'
    elif '_bad' in stem:
        return truth_root / plot / f'truth_{run}_bad.txt'
    else:
        return truth_root / plot / f'{run}.txt'


def get_reference_images(image_path: Path, ref_dir: Path = REF_DIR):
    """find_reference_images() only strips a trailing _run<N>; bad_images
    filenames end in _run<N>_bad[...], so strip that suffix first."""
    stem = image_path.stem
    for suffix in ('_bad_small', '_bad_medium', '_bad'):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    lookup_path = image_path.with_name(stem + image_path.suffix)
    return find_reference_images(lookup_path, ref_dir)


def existing_truth_path(
    plot: str,
    image_path: Path,
    truth_root: Path = TRUTH_ROOT,
    bad_image_root: Path = BAD_IMAGE_ROOT,
) -> 'Path | None':
    """Where find_truth_file() would look (truth/<plot>/...), plus a fallback
    to bad_images/<plot>/<same name> -- a handful of early bad-image truth
    files were saved colocated with their images instead of under truth/.
    New saves always go to the canonical truth_path_for() location."""
    canonical = truth_path_for(plot, image_path, truth_root)
    if canonical.exists():
        return canonical
    legacy = bad_image_root / plot / canonical.name
    if legacy.exists():
        return legacy
    return None


def build_worklist(
    instructions: dict,
    event_type: dict,
    image_root: Path = IMAGE_ROOT,
    bad_image_root: Path = BAD_IMAGE_ROOT,
    truth_root: Path = TRUTH_ROOT,
) -> pd.DataFrame:
    rows = []
    bad_pngs = [p for p in sorted(bad_image_root.rglob('*.png')) if '_bad' in p.stem]
    for png in sorted(image_root.rglob('*.png')) + bad_pngs:
        if '.ipynb_checkpoints' in png.parts:
            continue
        plot = png.parent.name
        m = re.search(r'run(\d+)', png.stem)
        run = int(m.group(1)) if m else None
        rows.append({
            'plot': plot,
            'image_path': png,
            'run': run,
            'event_type': event_type.get(run, 'unknown'),
            'truth_path': truth_path_for(plot, png, truth_root),
            'has_truth': existing_truth_path(plot, png, truth_root, bad_image_root) is not None,
        })
    return pd.DataFrame(rows).sort_values(['plot', 'run']).reset_index(drop=True)


def print_coverage(df: pd.DataFrame) -> None:
    """Displays the coverage table itself (rather than returning it) so
    calling this as the last line of a notebook cell doesn't also trigger
    Jupyter's automatic repr display of a returned DataFrame -- which would
    print the same table twice."""
    summary = df.groupby('plot')['has_truth'].agg(has_truth='sum', total='count')
    summary['missing'] = summary['total'] - summary['has_truth']
    display(summary)
    print(f'Overall: {df["has_truth"].sum()}/{len(df)} images have truth')


def _img_cell(path: Path, caption: str, width: int = 420) -> str:
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return (
        '<div style="text-align:center; margin:4px">'
        f'<div style="font-size:12px; margin-bottom:4px">{caption}</div>'
        f'<img src="data:image/png;base64,{data}" width="{width}">'
        '</div>'
    )


class TruthWriter:
    """Stateful driver: holds the worklist, navigation position, and the
    in-progress answer for whichever item is currently shown."""

    def __init__(
        self,
        image_root: Path = IMAGE_ROOT,
        bad_image_root: Path = BAD_IMAGE_ROOT,
        truth_root: Path = TRUTH_ROOT,
        ref_dir: Path = REF_DIR,
        plot_instructions_dir: Path = PLOT_INSTRUCTIONS_DIR,
        event_type_config: Path = EVENT_TYPE_CONFIG,
    ):
        self.image_root = image_root
        self.bad_image_root = bad_image_root
        self.truth_root = truth_root
        self.ref_dir = ref_dir

        self.instructions = load_instructions(plot_instructions_dir)
        self.event_type = load_event_types(event_type_config)
        self.items = build_worklist(
            self.instructions, self.event_type, image_root, bad_image_root, truth_root
        )

        self._plot_filter = None
        self._only_missing = True
        self._idx = 0

        self.quote = ''
        self.describe = ''
        self.compare = ''
        self.decision = 'good'

    # -- worklist / navigation state --------------------------------------

    def _current_view(self) -> pd.DataFrame:
        df = self.items
        if self._plot_filter is not None:
            df = df[df['plot'] == self._plot_filter]
        if self._only_missing:
            df = df[~df['has_truth']]
        return df.reset_index(drop=True)

    def _current_row(self):
        view = self._current_view()
        if view.empty or self._idx >= len(view):
            return None
        return view.iloc[self._idx]

    @property
    def current_truth_path(self) -> 'Path | None':
        """The exact path save() will write to for whatever is on screen now."""
        row = self._current_row()
        return None if row is None else row['truth_path']

    def print_coverage(self) -> None:
        print_coverage(self.items)

    def set_plot_filter(self, plot_name):
        """Restrict navigation to one plot (pass None to clear)."""
        self._plot_filter = plot_name
        self._idx = 0
        self.show()

    def only_missing(self, flag: bool = True):
        """True (default): only page through images without truth yet."""
        self._only_missing = flag
        self._idx = 0
        self.show()

    def goto(self, i: int):
        view = self._current_view()
        if not len(view):
            print('Nothing to show for the current filter.')
            return
        if not (0 <= i < len(view)):
            print(f'index out of range: 0..{len(view) - 1}')
            return
        self._idx = i
        self.show()

    def next_item(self):
        self.goto(self._idx + 1)

    def prev_item(self):
        self.goto(self._idx - 1)

    def jump_to(self, plot: str, run: int):
        view = self._current_view()
        matches = view[(view['plot'] == plot) & (view['run'] == run)]
        if matches.empty:
            print(f'No match for plot={plot!r} run={run} in the current view '
                  f'(try only_missing(False) if it might already have truth).')
            return
        self.goto(int(matches.index[0]))

    # -- display ------------------------------------------------------------

    def show(self):
        row = self._current_row()
        if row is None:
            print('Nothing to show (worklist is empty, or index out of range for the current filter).')
            return
        view = self._current_view()
        status = 'has truth -- editing' if row['has_truth'] else 'new'

        # 1. meta data
        print(f'[{self._idx + 1}/{len(view)}]  {row["plot"]}  run{row["run"]}  '
              f'event={row["event_type"]}  ({status})')
        print(f'truth file to write: {row["truth_path"]}')

        # 2. instruction
        print('\n--- Instruction ---')
        print(textwrap.fill(
            self.instructions.get(row['plot'], '(no instruction found for this plot)'),
            width=100,
        ))

        # 3. reference plot(s) + 4. input plot, side by side (input last/rightmost)
        refs = get_reference_images(row['image_path'], self.ref_dir)
        print(f'\n--- Reference plot(s) ({len(refs)}) alongside input plot ---')
        cells = ''.join(_img_cell(ref, ref.name) for ref in refs)
        cells += _img_cell(row['image_path'], f'INPUT: {row["image_path"].name}')
        display(HTML(f'<div style="display:flex; flex-wrap:wrap; gap:8px">{cells}</div>'))

        found = (
            existing_truth_path(row['plot'], row['image_path'], self.truth_root, self.bad_image_root)
            if row['has_truth'] else None
        )
        if found is not None:
            print(f'\n--- Existing truth ({found}) -- call blank_answer() to start fresh, or edit in place ---')
            print(found.read_text())
        else:
            print('\n--- No truth yet: call blank_answer(), fill in quote/describe/compare/decision, then save() ---')

    # -- answer / save --------------------------------------------------------

    def blank_answer(self):
        """Reset quote/describe/compare/decision for the currently shown item."""
        row = self._current_row()
        if row is None:
            print('Nothing to reset -- call show() first.')
            return
        self.quote = f'Since this is a "{row["event_type"]}" plot, the instruction is "..."'
        self.describe = ''
        self.compare = ''
        self.decision = 'good'
        print('Reset quote/describe/compare/decision -- edit them, then call save().')

    def save(self, overwrite: bool = False, advance: bool = True):
        row = self._current_row()
        if row is None:
            print('Nothing to save -- call show() first.')
            return

        truth_path = row['truth_path']
        found = existing_truth_path(row['plot'], row['image_path'], self.truth_root, self.bad_image_root)
        if found is not None and not overwrite:
            print(f'Truth already exists at {found} -- call save(overwrite=True) to replace it.')
            return

        text = (
            '- Quote the relevant section of instructions for the input plot\n'
            f'{self.quote}\n'
            ' - Describe the input plot\n'
            f'{self.describe}\n'
            ' - Compare input plot to the instruction\n'
            f'{self.compare}\n'
            ' - Decide if the plot is good or bad\n'
            f'The plot is {self.decision}.\n'
        )

        truth_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.write_text(text)
        print(f'Saved: {truth_path}')

        self.items.loc[
            (self.items['plot'] == row['plot']) & (self.items['image_path'] == row['image_path']),
            'has_truth',
        ] = True

        if advance:
            # Under only_missing=True the just-saved row drops out of the view,
            # so the next "new" item slides into the same _idx -- show(), not
            # next_item(), is what actually advances in that case.
            if self._only_missing:
                self.show()
            else:
                self.next_item()
