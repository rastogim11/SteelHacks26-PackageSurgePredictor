"""
Spoken weekly briefing.

    python -m briefing              print the narration
    python -m briefing --speak      also write outputs/briefing.mp3
    python -m briefing --scenario disposal_30d

Replaces the old elevenLabs.py scratch file.

The narration is built from a fixed template with numbers substituted in, not
written by a model. That keeps it auditable - what gets spoken can be traced
back to a row of week_plan.csv - and it keeps the project inside the No Wrapper
track, which the Readme commits to.

Power BI cannot call an external API from a button, so the intended wiring is
to run this in the nightly job and have a dashboard button open the file.

The API key is read from ELEVENLABS_API_KEY, via .env if present. It is never
written into this file. Without --speak no key is needed at all, so the text
can be checked before anything is synthesised.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from panther import config as cfg

OUT = Path(cfg.OUTPUT_DIR)
VOICE_ID = 'JBFqnCBsd6RMkjVDRZzb'


def _plural(n: float, one: str, many: str) -> str:
    return one if round(n) == 1 else many


def narration(scenario: str = 'current') -> str:
    """Build the spoken text from week_plan.csv."""
    path = OUT / 'week_plan.csv'
    if not path.exists():
        raise SystemExit(f'No {path}. Run: python -m panther.run predict')

    plan = pd.read_csv(path, parse_dates=['date', 'week_start', 'as_of'])
    wk = plan[plan['scenario'] == scenario]
    if wk.empty:
        raise SystemExit(f'No rows for scenario {scenario!r}. '
                         f'Available: {sorted(plan["scenario"].unique())}')

    week_of = wk['week_start'].min().strftime('%B %-d')
    per_site = wk.groupby('site').agg(peak_occ=('occupancy_hat', 'max'),
                                      lockers=('lockers', 'max'),
                                      hours=('total_hours', 'sum'),
                                      intake=('intake', 'sum'))
    per_site['util'] = per_site['peak_occ'] / per_site['lockers'] * 100

    lines = [f'Panther Post briefing for the week of {week_of}.']

    # Labour and holdings lead; raw intake is the least actionable number.
    total_hours = per_site['hours'].sum()
    lines.append(
        f'Across all mailrooms, {total_hours:.0f} labour hours are needed, '
        f'against a pool of {sum(cfg.HEADCOUNT_FTE.values()):.1f} full time staff.')

    # Over-capacity sites are the only ones worth naming aloud.
    over = per_site[per_site['util'] > 100].sort_values('util', ascending=False)
    if len(over):
        for site, r in over.iterrows():
            lines.append(
                f'{site} is over capacity: {r.peak_occ:,.0f} parcels held '
                f'against {r.lockers:,.0f} lockers, '
                f'{r.util:.0f} percent of capacity.')
    else:
        lines.append('No mailroom is forecast to exceed its locker capacity this week.')

    # The staffing call, at the single busiest site.
    top = per_site['hours'].idxmax()
    day_peak = wk[(wk['site'] == top) & wk['is_open']]
    if not day_peak.empty:
        d = day_peak.loc[day_peak['assign_fte'].idxmax()]
        lines.append(
            f'{top} needs the most cover: {d.assign_fte:g} '
            f'{_plural(d.assign_fte, "person", "people")} on {d.day_name}, '
            f'its heaviest day at {d.intake:.0f} parcels.')

    # The disposal finding, stated as a comparison rather than asserted.
    if scenario == 'current' and 'disposal_30d' in set(plan['scenario']):
        alt = plan[plan['scenario'] == 'disposal_30d'].groupby('site')['occupancy_hat'].max()
        cur = per_site['peak_occ']
        saved = (cur - alt).sort_values(ascending=False)
        site = saved.index[0]
        if saved.iloc[0] > 1:
            pct = (1 - alt[site] / cur[site]) * 100
            lines.append(
                f'A thirty day return to sender policy would cut {site} '
                f'from {cur[site]:,.0f} parcels held to {alt[site]:,.0f}, '
                f'a reduction of {pct:.0f} percent.')

    lines.append('Locker counts are provisional pending figures from Operations.')
    return ' '.join(lines)


def speak(text: str, out_path: Path) -> None:
    """Synthesise to mp3. Requires ELEVENLABS_API_KEY."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    key = os.environ.get('ELEVENLABS_API_KEY')
    if not key:
        raise SystemExit(
            'ELEVENLABS_API_KEY is not set.\n'
            '  Copy .env.example to .env and put your key in it, or:\n'
            '  export ELEVENLABS_API_KEY=your_key_here')

    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=key)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id='eleven_multilingual_v2',
        output_format='mp3_44100_128',
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        for chunk in audio:
            fh.write(chunk)
    print(f'  audio -> {out_path}')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='briefing')
    ap.add_argument('--scenario', default='current',
                    help='current (default) or disposal_30d')
    ap.add_argument('--speak', action='store_true',
                    help='synthesise audio as well as printing the text')
    ap.add_argument('--out', default=str(OUT / 'briefing.mp3'))
    args = ap.parse_args(argv)

    text = narration(args.scenario)
    print(f'\n{text}\n')
    if args.speak:
        speak(text, Path(args.out))


if __name__ == '__main__':
    main()
